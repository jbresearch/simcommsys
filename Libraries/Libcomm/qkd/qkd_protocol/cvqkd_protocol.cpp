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
cvqkd_protocol::split(libbase::vector<double>& measurements_alice,
                      libbase::vector<double>& measurements_bob,
                      int N_PE)
{

    // Setting size of N_PE for parameter estimation.
    this->N_PE = N_PE;

    assert(measurements_alice.size() == measurements_bob.size() &&
           "Alice and Bob's measurement vector sizes are not equal.");

    assert(N_PE > 0 && "N_PE must be > 0.");

    const int N = measurements_alice.size();

    libbase::vector<double> X_PE, Y_PE, X_raw, Y_raw;
    X_PE.init(N_PE);
    Y_PE.init(N_PE);
    X_raw.init(N - N_PE);
    Y_raw.init(N - N_PE);

    // First N_PE -> PE
    for (int i = 0; i < N_PE; ++i) {
        X_PE(i) = measurements_alice(i);
        Y_PE(i) = measurements_bob(i);
    }

    for (int i = N_PE; i < N; ++i) {
        const int j = i - N_PE;
        X_raw(j) = measurements_alice(
            i); // Unnormalised key of Alice to be used for post-processing
        Y_raw(j) = measurements_bob(
            i); // Unnormalised key of Bob to be used for post-processing
    }

    // Prints vectors just for debuggin.
    // TODO: To delete.
    print_vector("(Prints from cv_qkdprotocol.cpp) Prints Measurements values "
                 "of Alice: ",
                 measurements_alice);
    print_vector(
        "(Prints from cv_qkdprotocol.cpp) Prints Measurements values of Bob: ",
        measurements_bob);
    print_vector("(Prints from cv_qkdprotocol.cpp) Prints values of X_PE: ",
                 X_PE);
    print_vector("(Prints from cv_qkdprotocol.cpp) Prints values of Y_PE: ",
                 Y_PE);
    print_vector("(Prints from cv_qkdprotocol.cpp) Prints values of X_Raw: ",
                 X_raw);
    print_vector("(Prints from cv_qkdprotocol.cpp) Prints values of Y_raw: ",
                 Y_raw);

    return {X_PE, Y_PE, X_raw, Y_raw};
}

std::tuple<double, double, double>
cvqkd_protocol::parameter_estimation_optical_fiber(
    const libbase::vector<double>& X_PE,
    const libbase::vector<double>& Y_PE,
    int N_0,
    double v_el,
    double detector_efficiency)
{

    /**
     * Reference for parameter estimation equations:
     *   Chai, Geng, et al. "Parameter estimation of atmospheric
     * continuous-variable quantum key distribution." Physical Review A, 99(3),
     * 032326 (2019), Section A.
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
    double s_xx = 0.0;      // Σ x_i^2
    long double s_xy = 0.0; // Σ x_i y_i
    for (int i = 0; i < m; ++i) {
        const long double x = X_PE(i);
        const long double y = Y_PE(i);
        s_xx += x * x;
        s_xy += x * y;
    }

    // If all x_i are zero, t̂ is undefined.
    assert(s_xx > 0.0L && "PE eq: sum of x_i^2 is zero; t_hat undefined.");
    double t_hat = s_xy / s_xx;

    double sse = 0.0; // Σ (y_i − t̂ x_i)²
    for (int i = 0; i < m; ++i) {
        const long double resid = (Y_PE(i)) - t_hat * (X_PE(i));
        sse += resid * resid;
    }

    // Calculating sigma^2_hat which is an estimate of variance V_N
    double sigma2_hat = sse / m; // MLE (1/m)

    // Calculating sigma^2_0 which is σ^2_0 = N_0(1 + v_el)
    double sigma2_0 = N_0 * (1 + v_el);

    // Calculating estimate of epsilon: Epsilon_hat (eq. (5))
    Epsilon_hat = (sigma2_hat - sigma2_0) / (t_hat * N_0);
    std::cout << "(Prints from cvqkd_protocol.cpp) Epsilon_hat = "
              << Epsilon_hat << std::endl;

    if (Epsilon_hat < 0) {
        Epsilon_hat = 0; // Epsilon_hat cannot be negative.
    }

    std::cout << "(Prints from cvqkd_protocol.cpp) clipped Epsilon_hat = "
              << Epsilon_hat << std::endl;

    // Calculating estimate of transmittance: T_hat (eq. (5))
    // Note: I still need to add, v_el, N_0 and det_ff as serialized parameters
    // to the cv-qkd protocol for parameter estimation.
    T_hat = (t_hat * t_hat / detector_efficiency);
    std::cout << "(Prints from cvqkd_protocol.cpp) T_hat = " << T_hat
              << std::endl;

    // Calculating estimate for x_total_hat
    chi_total_hat = ((sigma2_hat) / (t_hat * t_hat)) - 1;
    std::cout << "(Prints from cvqkd_protocol.cpp) X_total_hat = " << T_hat
              << std::endl;

    return {T_hat, Epsilon_hat, chi_total_hat};
}

// Mutual Information for the GG02 protocol
double
cvqkd_protocol::calculate_mutual_information(double chi_total_hat, double VA)
{
    double I_AB = 0;
    double V = VA + 1;

    // Equation to calculate I_AB for homodyne detection and under collective
    // attacks.

    /*
     * References for I_AB calculation:
     *   [1] Lodewyck, Jérôme, et al. "Quantum key distribution over 25 km with
     * an all-fiber continuous-variable system." Physical Review A—Atomic,
     * Molecular, and Optical Physics 76.4 (2007): 042305. [2] Zhang, Y., Bian,
     * Y., Li, Z., Yu, S. and Guo, H., 2024. Continuous-variable quantum key
     * distribution system: Past, present, and future. Applied Physics Reviews,
     * 11(1).
     */

    I_AB = 0.5 * std::log2((V + chi_total_hat) /
                           (chi_total_hat + 1)); // In bits/pulse
    // I_AB_kbps = I_AB * repetition_rate;

    return I_AB;
}

// Holevo Bound for the GG02 protocol
double
cvqkd_protocol::calculate_holevo_bound(double V,
                                       double T_hat,
                                       double Epsilon_hat,
                                       double X_total_hat)
{
    double X_BE = 0; // X is chi
    double X_line_hat = 0;
    double X_hom_hat = 0;

    // Original equations of X_line and X_hom
    // X_hom = (1 - detector_efficiency + v_el)/detector_efficiency;
    // Xline = (1/T) - 1 + epsilon
    // Xtotal = Xline + (Xhom/T)

    /*
     * References for X_BE calculation:
     *   [1] Lodewyck, Jérôme, et al. "Quantum key distribution over 25 km with
     * an all-fiber continuous-variable system." Physical Review A—Atomic,
     * Molecular, and Optical Physics 76.4 (2007): 042305. [2] Zhang, Y., Bian,
     * Y., Li, Z., Yu, S. and Guo, H., 2024. Continuous-variable quantum key
     * distribution system: Past, present, and future. Applied Physics Reviews,
     * 11(1).
     */

    // Estimate X_line and X_hom from T_hat, Epsilon_hat and X_total_hat
    X_line_hat = (1 / T_hat) - 1 + Epsilon_hat;
    X_hom_hat = T_hat * (X_total_hat - X_line_hat);

    const double A = V * V * (1.0 - 2.0 * T_hat) + 2.0 * T_hat +
                     (T_hat * T_hat) * std::pow(V + X_line_hat, 2.0);
    const double B = (T_hat * T_hat) * std::pow(V * X_line_hat + 1.0, 2.0);

    const double sqrt_B = safe_sqrt(B);

    const double C_num = A * X_hom_hat + V * sqrt_B + T_hat * (V + X_line_hat);
    const double C_den = T_hat * (V + X_total_hat);

    if (C_den == 0.0) {
        throw std::invalid_argument(
            "compute_holevo_cvqkd: division by zero in C_den = T*(V+Xtot).");
    }
    const double C = C_num / C_den;

    const double D = sqrt_B * (V + sqrt_B * X_hom_hat) / C_den;

    const double disc1 = A * A - 4.0 * B;
    const double disc2 = C * C - 4.0 * D;

    const double lambda1 = std::sqrt(0.5) * safe_sqrt(A + safe_sqrt(disc1));
    const double lambda2 = std::sqrt(0.5) * safe_sqrt(A - safe_sqrt(disc1));
    const double lambda3 = std::sqrt(0.5) * safe_sqrt(C + safe_sqrt(disc2));
    const double lambda4 = std::sqrt(0.5) * safe_sqrt(C - safe_sqrt(disc2));

    // --- Holevo bound X_BE for homodyne detection with Gaussian modulated
    // coherent states.
    X_BE = bosonic_entropy_G((lambda1 - 1.0) / 2.0) +
           bosonic_entropy_G((lambda2 - 1.0) / 2.0) -
           bosonic_entropy_G((lambda3 - 1.0) / 2.0) -
           bosonic_entropy_G((lambda4 - 1.0) / 2.0);

    // X_BE_kbps = IBE * repetition_rate;
    return X_BE; // In bits/pulse.
}

double
cvqkd_protocol::compute_beta_mdr(double code_rate, double snr_linear)
{
    // beta is the reconciliation efficiency for MDR
    assert(code_rate >= 0.0);
    const double C = calculate_shannon_capacity_awgn(snr_linear);
    assert(C > 0.0 && "Capacity is zero (SNR too low) — cannot compute beta");

    const double beta = code_rate / C;

    assert(beta > 1 && "beta = R/C exceeded 1.0 — code rate above capacity or "
                       "wrong capacity model.");

    return beta;
}

const int
cvqkd_protocol::calculate_finite_size_effects_secret_key_length()
{
    /**
     * Finite-Size Secret Key Length Calculation (CV-QKD)
     *
     * Parameters:
     *   - n: Number of samples after parameter estimation (n = N - N_PE).
     *   - beta: Reconciliation efficiency, computed as beta = R /
     * C(SNR_linear), where C(SNR_linear) is the Shannon limit: C = 0.5 * (1 +
     * log2(SNR_linear)).
     *   - I_AB: Mutual information between Alice and Bob (bits/pulse).
     *           Use calculate_mutual_information().
     *   - chi_BE: Holevo bound between Bob and Eve for reverse reconciliation
     * (bits/pulse). Use calculate_holevo_bound().
     *   - delta(n): Finite-size security offset, ensuring key security against
     * statistical fluctuations in parameter estimation and error correction.
     *               See Lodewyck et al., Phys. Rev. A 76, 042305 (2007).
     *
     * References:
     *   1. Milicevic, M., Feng, C., Zhang, L.M., & Gulak, P.G. (2018).
     *      Quasi-cyclic multi-edge LDPC codes for long-distance quantum
     * cryptography. npj Quantum Information, 4(1), 21.
     *   2. Leverrier, A., Grosshans, F., & Grangier, P. (2010).
     *      Finite-size analysis of a continuous-variable quantum key
     * distribution. Phys. Rev. A, 81(6), 062343.
     */

    assert(n_samples > 0);
    assert(smoothing_parameter > 0);

    // Equation (32) from Reference 2
    double delta_n =
        7 * std::sqrt(std::log2(2 / smoothing_parameter) / n_samples);
    std::cout << "(prints from cvqkdprotocol.cpp) delta(n) = " << delta_n
              << std::endl;

    assert(beta_mdr > 0.0 && beta_mdr <= 1.0);
    assert(I_AB >= 0.0 && chi_BE >= 0.0);
    assert(delta_n >= 0);

    // Equation from reference 2

    // Finite-Size Effects Case.
    const double rate_per_pulse = (beta_mdr * I_AB) - chi_BE - delta_n;

    /* TODO: delete lines 270-271. For now in the initial tests I am excluding
     * delta(n) in line 266 and I used Beta = 1 as the framesize is too small
     * and beta is too small as well just for testing purposes.*/

    // // Asymptotic Case.
    // const double rate_per_pulse = (1 * I_AB) - chi_BE;

    assert(rate_per_pulse <= 0.0 && "Negative secret key rate/pulse!");

    // l = n[βIAB − χBE - delta(n)] from Reference 2
    int l = std::floor(n_samples * rate_per_pulse);

    // assert(l < 0 && "Computed length of secret key is negative!");
    if (l < 0) {
        l = 0;
    } // To be used to calculate final SKR in Results collector.

    // Question: Should I use the assert or if statement? And should I have
    // FER=1 if l = 0?
    return l;
}

// Returns final secret keys KA and KB.
std::pair<libbase::vector<bool>, libbase::vector<bool>>
cvqkd_protocol::postprocess(libbase::vector<double>&& alice_measurements,
                            libbase::vector<double>&& bob_measurements)
{
    // Instaniates both final secret keys KA and KB.
    libbase::vector<bool> final_secret_key_KA;
    libbase::vector<bool> final_secret_key_KB; //

    libbase::vector<double> X, Y;
    X.init(alice_measurements.size()); // alice_measurements == X_raw
    Y.init(bob_measurements.size());   // bob_measurements == Y_raw

    // Calculates L2 norms.
    double nX = l2(alice_measurements);
    double nY = l2(bob_measurements);
    assert(nX > 0.0 && nY > 0.0 && "cannot normalise a zero vector");

    // Normalises the X_raw and Y_raw measurement vectors of Alice and Bob to
    // get X and Y.
    for (int i = 0; i < X.size(); ++i)
        X(i) = alice_measurements(i) / nX;
    for (int i = 0; i < Y.size(); ++i)
        Y(i) = bob_measurements(i) / nY;

    print_vector(
        "(Prints from cv_qkdprotocol.cpp) Prints values of X (normalized): ",
        X);
    print_vector(
        "(Prints from cv_qkdprotocol.cpp) Prints values of Y (normalized): ",
        Y);

    print_vector("(Prints from cv_qkdprotocol.cpp) Prints Bob's Vector s: ",
                 bob_vector_s);

    // Generate Vector C from Bob's vector s.
    libbase::vector<int> encoded_int(get_codec_output_bits_n());

    libbase::vector<int> bob_vector_int(bob_vector_s.size());
    for (int i = 0; i < bob_vector_s.size(); ++i) {
        bob_vector_int(i) = bob_vector_s(i);
    }

    // Encodes Vector S of Bob to get Vector C.
    cdc->encode(bob_vector_int, encoded_int);

    bob_vector_c.init(encoded_int.size());

    // Obtains Bob'c vector C of bool type through conversion.
    for (int i = 0; i < encoded_int.size(); ++i) {
        bob_vector_c(i) = encoded_int(i);
    }

    print_vector(
        "(Prints from cv-qkdprotocol.cpp) (Encoded Result) Vector C : ",
        bob_vector_c);

    /* Modulation step: Generate Vector M using BPSK modulation.*/

    // The direct_block_informed_embedder uses the embed method from the base
    // class block_informed_embedder.h.

    // Data to embed which is encoded bit vector C, converted from bool to int.
    libbase::vector<int> data_to_embed(bob_vector_c.size());
    for (int i = 0; i < bob_vector_c.size(); ++i) {
        data_to_embed(i) = bob_vector_c(i);
    }

    // Vector M which stores the modulated signal.
    libbase::vector<double> vector_M;
    vector_M.init(get_codec_output_bits_n());

    embedder->set_blocksize(Y.size());

    embedder->embed(alphabet_size, data_to_embed, Y, vector_M);

    print_vector(
        "(Prints from cv-qkdprotocol.cpp) Modulated Vector M (from embedder):",
        vector_M);

    /* Demodulation step to get the Probability Table for the decoder. */

    // Instantiate the AWGN channel object.
    demodulation_channel = std::make_shared<libcomm::awgn1d>();

    std::cout << "\n(Prints from cv-qkdprotocol.cpp) Description of "
                 "Demodulation channel: "
              << demodulation_channel->description() << std::endl;

    std::cout
        << "\n(Prints from cv-qkdprotocol.cpp) Print value of SNR_linear: "
        << SNR_linear << std::endl;

    // Convert SNR to dB
    double SNR_dB = 10.0 * std::log10(SNR_linear);

    std::cout << "\n(Prints from cv-qkdprotocol.cpp) SNR (dB): " << SNR_dB
              << std::endl;

    // Set SNR_db in AWGN channel
    demodulation_channel->set_parameter(SNR_dB);

    // Instantiate Probability Table.
    libbase::vector<libbase::vector<double>> prob_table;

    std::cout << "\n(Prints from cv-qkdprotocol.cpp) Size of Vector M = "
              << vector_M.size() << "\tSize of Vector X = " << X.size()
              << std::endl;

    // Perform Demodulation to get Probability Table.
    embedder->extract(*demodulation_channel, vector_M, X, prob_table);

    std::cout << "\n(Prints probability table from cv-qkdprotocol.cpp:)"
              << std::endl;
    print_prob_table(prob_table);

    /*LDPC decoding using the prob_table to get vector s_hat */
    cdc->init_decoder(prob_table);

    auto decoded = libbase::vector<int>();
    cdc->decode(decoded);

    print_vector("\n(Prints from cv-qkdprotocol.cpp using cout:) Decoded bits:",
                 decoded);

    // Copy decoded bits to vector s_hat  of Alice which is of bool type.
    libbase::vector<bool> vector_s_hat(get_codec_input_bits_k());

    for (int i = 0; i < decoded.size(); ++i) {
        vector_s_hat(i) = decoded(i);
    }

    print_vector("\n(Prints from cv-qkdprotocol.cpp) Vector S_hat of Alice:",
                 vector_s_hat);

    std::cout << "\n(Prints from cv-qkdprotocol.cpp) Hamming distance between "
                 "vectors s and s_hat: "
              << libbase::hamming(bob_vector_s, vector_s_hat) << std::endl;

    /* Calculating Hashing for Vectors s and s_hat */
    std::uint32_t hash_hs = crc32_ieee<>::compute(bob_vector_s);
    std::cout << "Hash hs of Bob's vector s = " << hash_hs << std::endl;

    std::uint32_t hash_hs_hat = crc32_ieee<>::compute(vector_s_hat);
    std::cout << "Hash hs_hat of Alice's vector s_hat = " << hash_hs_hat
              << std::endl;

    if (hash_hs == hash_hs_hat) {
        H_check = 1;
        std::cout << "(Prints from cv-qkdprotocol.cpp) Hash Check = " << H_check
                  << std::endl;

        /* Calculate Beta for MDR: beta = R/C(S) taken from the Quasi Cyclic
         * Paper 2018, Mario Milicevic. C(S) is the Shannon Capacity of an AWGN
         * channel. */

        double R_code = cdc->rate();

        std::cout << "\n(Prints from cv-qkdprotocol.cpp) R_code  = " << R_code
                  << std::endl;

        double C_awgn_capacity = calculate_shannon_capacity_awgn(SNR_linear);

        beta_mdr = R_code / C_awgn_capacity;
        std::cout << "\n(Prints from cv-qkdprotocol.cpp) beta_mdr  = "
                  << beta_mdr << std::endl;

        /* Calculate length l of final secret key */
        std::cout << "\nTesting equation that calculates final length l of "
                     "secret key (prints from cvqkd_protocol.cpp)"
                  << std::endl;

        int len_secret_key = calculate_finite_size_effects_secret_key_length();
        std::cout << "\n (prints from cvqkd_protocol.cpp) Length l of final "
                     "secret key = "
                  << len_secret_key << std::endl;

        // Sets length of secret keys KA and KB to later be able to retrieve
        // them for the results collector.
        libbase::vector<bool> final_secret_key_KA(len_secret_key);
        libbase::vector<bool> final_secret_key_KB(len_secret_key); //

        /* Perform Privacy Amplification */

        // // Intialise Privacy Amplification system.
        // pa_system = std::make_shared<libcomm::pa_standard_toeplitz<bool>>();

        std::cout << "\n (prints from cvqkd_protocol.cpp) Privacy "
                     "Amplification System Description = "
                  << pa_system.description() << std::endl;

        // to use alphabet size of 2
        pa_system.set_alphabet_size(alphabet_size);
        // Length of final key after doing PA.
        pa_system.set_L(len_secret_key);
        // Length of pre-hased key which in this case is the size of vectors s
        // and s_hat.
        pa_system.set_N(get_codec_input_bits_k());

        int starting_vector_len = pa_system.generate_starting_vector_length();

        // Printing PA System Parameters
        std::cout
            << "(prints from cvqkd_protocol.cpp)  Length of starting vector = "
            << starting_vector_len << std::endl;

        std::cout
            << "(prints from cvqkd_protocol.cpp)  Length L of the PA system: "
            << pa_system.get_L() << std::endl;
        std::cout
            << "(prints from cvqkd_protocol.cpp)  Length N of the PA system: "
            << pa_system.get_N() << std::endl;
        std::cout << "(prints from cvqkd_protocol.cpp)  Alphabet size of the "
                     "PA system: "
                  << pa_system.get_alphabet_size() << std::endl;

        // Generate starting vector.
        libbase::vector<bool> starting_vector =
            pa_system.generate_starting_vector(starting_vector_len,
                                               pa_system.get_alphabet_size());

        // Generate Standard Toeplitz matrix.
        libbase::matrix<bool> standard_toeplitz_matrix =
            pa_system.generate_toeplitz_matrix(starting_vector);

        // Generates KB of Bob.
        final_secret_key_KB =
            pa_system.compute_hashed_key(standard_toeplitz_matrix,
                                         bob_vector_s,
                                         len_secret_key,
                                         bob_vector_s.size(),
                                         alphabet_size);

        // Generates KA of Alice.
        final_secret_key_KA =
            pa_system.compute_hashed_key(standard_toeplitz_matrix,
                                         vector_s_hat,
                                         len_secret_key,
                                         vector_s_hat.size(),
                                         alphabet_size);

        print_vector(
            "(prints from cvqkd_protocol.cpp)  Final Secret Key KA of Alice: ",
            final_secret_key_KA);
        print_vector(
            "(prints from cvqkd_protocol.cpp)  Final Secret Key KB of Bob: ",
            final_secret_key_KB);

        return {std::move(final_secret_key_KA), std::move(final_secret_key_KB)};
    } else {
        H_check = 0;
        std::cout << "(Prints from cv-qkdprotocol.cpp) Hash Check = " << H_check
                  << std::endl;

        int len_secret_key = 0;

        // Sets length of secret keys KA and KB to zero to later be able to
        // retrieve it for the results collector.
        final_secret_key_KA(len_secret_key);
        final_secret_key_KB(len_secret_key);

        return {std::move(final_secret_key_KA), std::move(final_secret_key_KB)};
    }
}

// Returns description of the protocol
std::string
cvqkd_protocol::description() const
{
    return "CV-QKD Protocol using the GG02 protocol with GM Coherent states";
}

//! Serialize protocol
std::ostream&
cvqkd_protocol::serialize(std::ostream& sout) const
{
    sout << "# Shot Noise Variance N_0" << std::endl;
    sout << N_0 << std::endl;
    sout << "# Electric Noise v_el" << std::endl;
    sout << v_el << std::endl;
    sout << "# Detector Efficiency eta" << std::endl;
    sout << detector_efficiency << std::endl;
    // Smoothing parameter bar epsilon which is used to calculate the final
    // length of the secret key.
    sout << "# Smoothing Parameter" << std::endl;
    sout << smoothing_parameter << std::endl;
    sout << "# Alphabet size" << std::endl;
    sout << alphabet_size << std::endl;
    sout << "# Codec" << std::endl;
    sout << cdc << std::endl;
    sout << "# Embedder" << std::endl;
    sout << embedder << std::endl;
    return sout;
}

//! Deserialize protocol
std::istream&
cvqkd_protocol::serialize(std::istream& sin)
{
    assertalways(sin.good());
    sin >> libbase::eatcomments >> N_0 >> libbase::verify;
    sin >> libbase::eatcomments >> v_el >> libbase::verify;
    sin >> libbase::eatcomments >> detector_efficiency >> libbase::verify;
    sin >> libbase::eatcomments >> smoothing_parameter >> libbase::verify;
    sin >> libbase::eatcomments >> alphabet_size >> libbase::verify;
    sin >> libbase::eatcomments >> cdc >> libbase::verify;
    sin >> libbase::eatcomments >> embedder >> libbase::verify;

    return sin;
}

const serializer cvqkd_protocol::shelper("qkd_protocol",
                                         "cvqkd_protocol",
                                         cvqkd_protocol::create);

} // namespace libcomm