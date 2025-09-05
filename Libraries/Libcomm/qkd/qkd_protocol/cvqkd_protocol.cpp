#include "cvqkd_protocol.h"
#include "codec/ldpc.h"
#include "gf.h"
#include <sstream>

using libbase::serializer;

namespace libcomm
{
    // Split fn to be used for parameter estimation and post-processing.
    std::tuple<libbase::vector<double>, // X_PE for Alice
    libbase::vector<double>, // Y_PE for Bob
    libbase::vector<double>, // Alice's raw key
    libbase::vector<double>> // Bob's raw key
    cvqkd_protocol::split(libbase::vector<double>& measurements_alice, libbase::vector<double>& measurements_bob,
    int N_PE)
    {
        assert(measurements_alice.size() == measurements_bob.size() && "Alice and Bob's measurement vector sizes are not equal.");

        assert(N_PE > 0 && "N_PE must be > 0.");

        const int N = measurements_alice.size();
        if (N_PE < 0) N_PE = 0;
        if (N_PE > N) N_PE = N;

        libbase::vector<double> X_PE, Y_PE, X_raw, Y_raw;
        X_PE.init(N_PE);
        Y_PE.init(N_PE);
        X_raw.init(N - N_PE);
        Y_raw.init(N - N_PE);

        // First N_PE -> PE
        for (int i = 0; i < N_PE; ++i) {
            X_PE(i) = measurements_alice(i);
            Y_PE(i)   = measurements_bob(i);
        }

        for (int i = N_PE; i < N; ++i) {
            const int j = i - N_PE;
            X_raw(j) = measurements_alice(i); // Unnormalised key of Alice to be used for post-processing
            Y_raw(j)   = measurements_bob(i); // Unnormalised key of Bob to be used for post-processing
        }

        return {X_PE, Y_PE, X_raw, Y_raw};
    }

    std::tuple<double, double, double> cvqkd_protocol::parameter_estimation_optical_fiber(libbase::vector<double>& X_PE, libbase::vector<double>& Y_PE, int N_0, double v_el, double detector_efficiency)
    {

        /* Reference of equations used to calculate the parameter estimation: Section A. of Chai, Geng, et al. "Parameter estimation of atmospheric continuous-variable quantum key distribution." Physical Review A 99.3 (2019): 032326.
        */

        assert(X_PE.size() == Y_PE.size() && "X_PE and Y_PE must have same size.");

        const int m = X_PE.size();
        assert(m > 0 && "sample size m must be > 0.");
        assert(N_0 > 0 && "shot noise N_0 must be > 0.");
        assert(v_el > 0 && "electric noise v_el must be > 0.");

        double T_hat = 0.0;
        double Epsilon_hat = 0.0;
        double chi_total_hat = 0.0;

        // Calculating t_hat (eq. (3)) where t = √ηT ∈ R
        double s_xx = 0.0;  // Σ x_i^2
        long double s_xy = 0.0;  // Σ x_i y_i
        for (int i = 0; i < m; ++i) {
        const long double x = X_PE(i);
        const long double y = Y_PE(i);
        s_xx += x * x;
        s_xy += x * y;
        }

        // If all x_i are zero, t̂ is undefined.
        assert(s_xx > 0.0L && "PE eq: sum of x_i^2 is zero; t_hat undefined.");
        double t_hat = s_xy / s_xx;


        double sse = 0.0;   // Σ (y_i − t̂ x_i)²
        for (int i = 0; i < m; ++i) {
            const long double resid = (Y_PE(i)) - t_hat * (X_PE(i));
            sse += resid * resid;
        }

        // Calculating sigma^2_hat which is an estimate of variance V_N
        double sigma2_hat = sse / m; // MLE (1/m)

        // Calculating sigma^2_0 which is σ^2_0 = N_0(1 + v_el)
        double sigma2_0 = N_0*(1 + v_el);

        // Calculating estimate of epsilon: Epsilon_hat (eq. (5))
        Epsilon_hat = (sigma2_hat - sigma2_0)/(t_hat*N_0);
        std::cout << "(Prints from cvqkd_protocol.cpp) Epsilon_hat = " << Epsilon_hat << std::endl;

        if(Epsilon_hat < 0)
        {
            Epsilon_hat = 0; // Epsilon_hat cannot be negative.
        }

        std::cout << "(Prints from cvqkd_protocol.cpp) clipped Epsilon_hat = " << Epsilon_hat << std::endl;

        // Calculating estimate of transmittance: T_hat (eq. (5))
        // Note: I still need to add, v_el, N_0 and det_ff as serialized parameters to the cv-qkd protocol for parameter estimation.
        T_hat = (t_hat*t_hat/detector_efficiency);
        std::cout << "(Prints from cvqkd_protocol.cpp) T_hat = " << T_hat << std::endl;


        // Calculating estimate for x_total_hat
        chi_total_hat = ((sigma2_hat)/(t_hat * t_hat)) - 1;
        std::cout << "(Prints from cvqkd_protocol.cpp) X_total_hat = " << T_hat << std::endl;

        return {T_hat, Epsilon_hat, chi_total_hat};
    }

    // Mutual Information for the GG02 protocol
    double cvqkd_protocol::calculate_mutual_information(double chi_total_hat, double VA)
    {
        double I_AB = 0;
        double V = VA + 1;

        // Equation to calculate I_AB for homodyne detection and under collective attacks.

        /* References for I_AB calculation: [1] Lodewyck, Jérôme, et al. "Quantum key distribution over 25 km with an all-fiber continuous-variable system." Physical Review A—Atomic, Molecular, and Optical Physics 76.4 (2007): 042305.
        [2] Zhang, Y., Bian, Y., Li, Z., Yu, S. and Guo, H., 2024. Continuous-variable quantum key distribution system: Past, present, and future. Applied Physics Reviews, 11(1).
        */

        I_AB = 0.5*std::log2((V + chi_total_hat)/(chi_total_hat + 1)); // In bits/pulse
        // I_AB_kbps = I_AB * repetition_rate;

        return I_AB;
    }

    // Holevo Bound for the GG02 protocol
    double cvqkd_protocol::calculate_holevo_bound(double V, double T_hat, double Epsilon_hat, double X_total_hat)
    {
        double X_BE = 0; // X is chi
        double X_line_hat = 0;
        double X_hom_hat = 0;

        // Original equations of X_line and X_hom
        // X_hom = (1 - detector_efficiency + v_el)/detector_efficiency;
        // Xline = (1/T) - 1 + epsilon
        // Xtotal = Xline + (Xhom/T)

        /* References for X_BE calculation: [1] Lodewyck, Jérôme, et al. "Quantum key distribution over 25 km with an all-fiber continuous-variable system." Physical Review A—Atomic, Molecular, and Optical Physics 76.4 (2007): 042305.
        [2] Zhang, Y., Bian, Y., Li, Z., Yu, S. and Guo, H., 2024. Continuous-variable quantum key distribution system: Past, present, and future. Applied Physics Reviews, 11(1).
        */

        // Estimate X_line and X_hom from T_hat, Epsilon_hat and X_total_hat
        X_line_hat = (1/T_hat) - 1 + Epsilon_hat;
        X_hom_hat = T_hat*(X_total_hat - X_line_hat);

        const double A = V*V * (1.0 - 2.0*T_hat) + 2.0*T_hat + (T_hat*T_hat) * std::pow(V + X_line_hat, 2.0);
        const double B = (T_hat*T_hat) * std::pow(V*X_line_hat + 1.0, 2.0);

        const double sqrt_B = safe_sqrt(B);

        const double C_num = A * X_hom_hat + V * sqrt_B + T_hat * (V + X_line_hat);
        const double C_den = T_hat * (V + X_total_hat);

        if (C_den == 0.0) {
            throw std::invalid_argument("compute_holevo_cvqkd: division by zero in C_den = T*(V+Xtot).");
        }
        const double C = C_num / C_den;

        const double D = sqrt_B * (V + sqrt_B * X_hom_hat) / C_den;

        const double disc1 = A*A - 4.0*B;
        const double disc2 = C*C - 4.0*D;

        const double lambda1 = std::sqrt(0.5) * safe_sqrt(A + safe_sqrt(disc1));
        const double lambda2 = std::sqrt(0.5) * safe_sqrt(A - safe_sqrt(disc1));
        const double lambda3 = std::sqrt(0.5) * safe_sqrt(C + safe_sqrt(disc2));
        const double lambda4 = std::sqrt(0.5) * safe_sqrt(C - safe_sqrt(disc2));

        // --- Holevo bound X_BE for homodyne detection with Gaussian modulated coherent states.
        X_BE =
            bosonic_entropy_G((lambda1 - 1.0) / 2.0) +
            bosonic_entropy_G((lambda2 - 1.0) / 2.0) -
            bosonic_entropy_G((lambda3 - 1.0) / 2.0) -
            bosonic_entropy_G((lambda4 - 1.0) / 2.0);

        // X_BE_kbps = IBE * repetition_rate;
        return X_BE; // In bits/pulse.
    }

    double cvqkd_protocol::compute_beta_mdr(double code_rate, double snr_linear)
    {
        // beta is the reconciliation efficiency for MDR
        assert(code_rate >= 0.0);
        const double C = calculate_shannon_capacity_awgn(snr_linear);
        assert(C > 0.0 && "Capacity is zero (SNR too low) — cannot compute beta");

        const double beta = code_rate / C;

        assert(beta >1 &&
            "beta = R/C exceeded 1.0 — code rate above capacity or wrong capacity model.");

        return beta;
    }

    const int cvqkd_protocol::calculate_finite_size_effects_secret_key_length()
    {
        /*
        n - is the number of samples after parameter estimation i.e. n = N - N_PE

        beta - is the reconcilation efficiency which is calculated using beta = R/C(SNR_linear) where C(SNR_linear) is the Shannon limit calculated using C(SNR_linear) = 0.5(1+log_2(SNR_linear)).

        I_AB - is the mutual information between Alice and Bob. Use calculate_mutual_information method to compute this. Unit is in bits/pulse.

        \chi_BE - is the Holevo bound between Bob and Eve for reverse reconciliation. Use calculate_holevo_bound to compute this. Units is in bits/pulse.

        delta (n) is the finite size offset term which is related to the security of privacy amplification. It is used to ensure that the key is secure even in the presence of statistical fluctuations in parameter estimation and error correction~\cite{lodewyck2007quantum}.

        Reference 1 for beta_mdr:
        Milicevic, M., Feng, C., Zhang, L.M. and Gulak, P.G., 2018. Quasi-cyclic multi-edge LDPC codes for long-distance quantum cryptography. npj Quantum Information, 4(1), p.21.

        Reference 2 for equations to calculate the offset size Delta(n) and thee length l of the final secret key:
        Leverrier, A., Grosshans, F. and Grangier, P., 2010. Finite-size analysis of a continuous-variable quantum key distribution. Physical Review A—Atomic, Molecular, and Optical Physics, 81(6), p.062343.
        */

        assert(n_samples > 0);
        assert(smoothing_parameter > 0);

        // Equation (32) from Reference 2
        double delta_n = 7 * std::sqrt(std::log2(2/smoothing_parameter)/n_samples);
        std::cout << "(prints from cvqkdprotocol.cpp) delta(n) = " << delta_n << std::endl;

        assert(beta_mdr > 0.0 && beta_mdr <= 1.0);
        assert(I_AB >= 0.0 && chi_BE >= 0.0);
        assert(delta_n >= 0);

        // Equation from reference 2.
        const double rate_per_pulse = (beta_mdr * I_AB) - chi_BE - delta_n;
        assert(rate_per_pulse <= 0.0 &&
            "Negative secret key rate/pulse!");

        // l = n[βIAB − χBE - delta(n)] from Reference 2
        int l = std::floor(n_samples * rate_per_pulse);

        // assert(l < 0 && "Computed length of secret key is negative!");
        if (l < 0) {l = 0;} // To be used to calculate final SKR in Results collector.

        // Question: Should I use the assert or if statement? And should I have FER=1 if l = 0?

        return l;
    }

    // Build an LDPC(gf2,double) from a config stream and install it into `cdc`.
    static std::shared_ptr<codec<libbase::vector>>
    make_ldpc_from_stream(std::istream& sin) {
        // ldpc<gf2,double> inherits codec<libbase::vector,double>  :contentReference[oaicite:3]{index=3}
        auto ldpc_ptr = std::make_shared<libcomm::ldpc<libbase::gf2,double>>();
        ldpc_ptr->serialize(sin); // loads v5/v6 format like in your snippet       :contentReference[oaicite:4]{index=4}
        return ldpc_ptr;
    }

    // Returns final secret key.
    libbase::vector<bool> cvqkd_protocol::postprocess(libbase::vector<double>&& alice_measurements,  libbase::vector<double>&& bob_measurements)
    {
        libbase::vector<bool> final_key;
        libbase::vector<double> X, Y;
        X.init(alice_measurements.size()); // alice_measurements == X_raw
        Y.init(bob_measurements.size()); // bob_measurements == Y_raw

        // Calculates L2 norms.
        double nX = l2(alice_measurements);
        double nY = l2(bob_measurements);
        assert(nX > 0.0 && nY > 0.0 && "cannot normalise a zero vector");

        // Normalises the X_raw and Y_raw measurement vectors of Alice and Bob to get X and Y.
        for (int i = 0; i < X.size(); ++i) X(i) = alice_measurements(i) / nX;
        for (int i = 0; i < Y.size(); ++i) Y(i) = bob_measurements(i) / nY;

        /* Bob: Randomly generate vector s. -> STILL TO DO */
        // const int k = 1000; // Size of information bits without encoding. // Still to define in an automated way -> probably to serialized related to the LDPC.

        // Vector s should be bool but I kept int due to future LDPC computations.
        // Still to randomly generate using libbase::randgen.

        int k = 3; // Size of vector s. Still need to change this depending from where I am calling the codec.
        libbase::vector<bool> s = create_vector_s(rng, k);

        std::cout << "\n (prints from cvqkd_protocol.cpp) Generated vector s [size k = " << k << "]: [";
        for (int i = 0; i < k; ++i) {
            if (i) std::cout << ", ";
            std::cout << s(i);
        }
        std::cout << "]\n\n";

        // // These parameters cannot be hard coded.
        // int len_secret_key;
        // double snr_linear = 0.0283;
        // double R_code = 0.02;
        // double beta_mdr = compute_beta_mdr(R_code, snr_linear);
        // std::cout << "\n (prints from cvqkd_protocol.cpp) Reconciliation Efficiency Beta MDR = " << beta_mdr << std::endl;

        // int n = 99000; // STILL TO DO: Samples left after PE with N=110k and N_PE = 11K. Need a getter to get the number of generated coherent states - N_PE.

        // // STILL TO DO: Same applies for I_AB and X_BE need to somehow get them into processing. Currently I am calculating these from qkd_commsys.h.
        // double I_AB = 1.04825;
        // double X_BE = 0.898772;
        // double E_PA = 1e-10; // privacy amplification failure probability per block. Probably has to be a serialized parameter in the cv_qkd_protocol.
        // int security_parameter = static_cast<int>(std::ceil(-std::log2(E_PA)));
        // std::cout << "\n (prints from cvqkd_protocol.cpp)  (number of states to be deduced for PA.) Security parameter s = " << security_parameter << std::endl;

        // // Still need to add setter fn for Iab , chbe, n_samples and beta_mdr this is to be added in the fullcycle method of qkd commsys before calling the post processing method.

        /* Setting codec of CV-QKD protocol. This will have to be refactored during code review with Johann.*/

        if (!cdc) {
            std::stringstream ss;
            ss <<
            "# Version\n"
            "5\n"
            "# SPA type (trad|gdl)\n"
            "gdl\n"
            "# Number of iterations\n"
            "50\n"
            "# Clipping method\n"
            "zero\n"
            "# Value of almostzero\n"
            "1e-100\n"
            "# Reduce generator matrix to REF? (true|false)\n"
            "1\n"
            "# Length (n)\n"
            "7\n"
            "# Dimension (m)\n"
            "7\n"
            "# Max column weight\n"
            "3\n"
            "# Max row weight\n"
            "3\n"
            "# Non-zero values (ones|random|provided)\n"
            "ones\n"
            "# Column weight vector\n"
            "7\n"
            "3 3 3 3 3 3 3\n"
            "# Row weight vector\n"
            "7\n"
            "3 3 3 3 3 3 3\n"
            "# Non zero positions per col\n"
            "3\n"
            "1 5 7\n"
            "3\n"
            "1 2 6\n"
            "3\n"
            "2 3 7\n"
            "3\n"
            "1 3W 4\n"
            "3\n"
            "2 4 5\n"
            "3\n"
            "3 5 6\n"
            "3\n"
            "4 6 7\n";

            /* Sets the codec I want to use in the CV-QKD protocol. This has to be changed in the code review and it has to go in the serialization ,method not here. */
            set_cvqkd_protocol_codec(make_ldpc_from_stream(ss));
        }

        std::cout << "(printing from cv-qkd-protocol.cpp the system details of the codec used in the cv-qkd protocol: " << cdc->description() << "\n";

        std::cout << "(printing from cv-qkd-protocol.pp) input bits k of codec =  " << get_codec_input_bits_k() << "\n";

        std::cout << "\nTesting equation that calculates final length l of secret key (prints from cvqkd_protocol.cpp)" << std::endl;

        int len_secret_key = calculate_finite_size_effects_secret_key_length();
        std::cout << "\n (prints from cvqkd_protocol.cpp) Length l of final secret key = " << len_secret_key << std::endl;

        // Sets length of secret key to later be able to retrieve it for the results collector.
        set_length_secret_key(len_secret_key);

        final_key.init(len_secret_key);

        return final_key;
    }


    // Returns description of the protocol
    std::string cvqkd_protocol::description() const { return "CV-QKD Protocol using the GG02 protocol with GM Coherent states";}

    //! Serialize protocol
    std::ostream& cvqkd_protocol::serialize(std::ostream& sout) const
    {
        sout << "# Number of samples for Parameter Estimation N_PE" << std::endl;
        sout << N_PE << std::endl;
        sout << "# Shot Noise Variance N_0" << std::endl;
        sout << N_0 << std::endl;
        sout << "# Electric Noise v_el" << std::endl;
        sout << v_el << std::endl;
        sout << "# Detector Efficiency eta" << std::endl;
        sout << detector_efficiency << std::endl;
        // Smoothing parameter bar epsilon which is used to calculate the final length of the secret key.
        sout << "# Smoothing Parameter" << std::endl;
        sout << smoothing_parameter << std::endl;
        // sout << "## Codec" << std::endl;
        // sout << cdc << std::endl;
        return sout;
    }

    //! Deserialize protocol
    std::istream& cvqkd_protocol::serialize(std::istream& sin)
    {
        assertalways(sin.good());
        sin >> libbase::eatcomments >> N_PE >> libbase::verify;
        sin >> libbase::eatcomments >> N_0 >> libbase::verify;
        sin >> libbase::eatcomments >> v_el >> libbase::verify;
        sin >> libbase::eatcomments >> detector_efficiency >> libbase::verify;
        sin >> libbase::eatcomments >> smoothing_parameter >> libbase::verify;
        // sin >> libbase::eatcomments >> cdc >> libbase::verify;

        return sin;
    }

    const serializer cvqkd_protocol::shelper("qkd_protocol", "cvqkd_protocol", cvqkd_protocol::create);

} // namespace libcomm