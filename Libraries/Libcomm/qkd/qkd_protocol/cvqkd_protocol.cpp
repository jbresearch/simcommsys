#include "cvqkd_protocol.h"
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

        if(Epsilon_hat < 0)
        {
            Epsilon_hat = 0; // Epsilon_hat cannot be negative.
        }

        // Calculating estimate of transmittance: T_hat (eq. (5))
        // Note: I still need to add, v_el, N_0 and det_ff as serialized parameters to the cv-qkd protocol for parameter estimation.
        T_hat = (t_hat*t_hat/detector_efficiency);

        // Calculating estimate for x_total_hat
        chi_total_hat = ((sigma2_hat)/(t_hat * t_hat)) - 1;

        return {T_hat, Epsilon_hat, chi_total_hat};
    }

    // Mutual Information for the GG02 protocol
    double cvqkd_protocol::calculate_mutual_information(double chi_total_hat, double VA)
    {
        double I_AB = 0;
        double V = VA + 1;

        // Equation to calculate I_AB for homodyne detection and under collective attacks.

        /* References for I_AB calculation: 1. Lodewyck, Jérôme, et al. "Quantum key distribution over 25 km with an all-fiber continuous-variable system." Physical Review A—Atomic, Molecular, and Optical Physics 76.4 (2007): 042305.
        2) Zhang, Y., Bian, Y., Li, Z., Yu, S. and Guo, H., 2024. Continuous-variable quantum key distribution system: Past, present, and future. Applied Physics Reviews, 11(1).
        */

        I_AB = 0.5*std::log2((V + chi_total_hat)/(chi_total_hat + 1)); // I_AB returned is in bits/pulse. To convert to kbps multiply with the repetition rate e.g. 350 kHz

        return I_AB;
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
        return sin;
    }

    const serializer cvqkd_protocol::shelper("qkd_protocol", "cvqkd_protocol", cvqkd_protocol::create);

} // namespace libcomm