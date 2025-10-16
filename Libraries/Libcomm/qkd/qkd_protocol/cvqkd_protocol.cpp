#include "cvqkd_protocol.h"
#include "codec/ldpc.h"
#include "gf.h"
#include "source/quantum_gaussian_source.h"
#include <cmath>
#include <sstream>

using libbase::serializer;

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

void
cvqkd_protocol::init(source<gaussian_state>& src_gen_base,
                     const std::shared_ptr<quantum_channel>& bob_channel)
{
    // Safely cast the base class reference to the derived class we need.
    auto& src_gen = dynamic_cast<quantum_gaussian_source&>(src_gen_base);

    // Now, correctly get the modulation variance from the source.
    m_modulation_variance = src_gen.get_VA();

    // Gets Bobs quantum channel from qkd_commsys.
    this->m_bob_channel = bob_channel;
}

// Split fn to be used for parameter estimation and post-processing.
std::tuple<libbase::vector<double>, // X_PE for Alice
           libbase::vector<double>, // Y_PE for Bob
           libbase::vector<double>, // Alice's raw key
           libbase::vector<double>> // Bob's raw key
cvqkd_protocol::split(libbase::vector<double>& alice_measurements,
                      libbase::vector<double>& bob_measurements)
{
    assert(alice_measurements.size() == bob_measurements.size() &&
           "Alice and Bob's measurement vector sizes are not equal.");

    assert(N_PE > 0 && "N_PE must be > 0.");

    const int N = alice_measurements.size();

    libbase::vector<double> X_PE, Y_PE, X_raw, Y_raw;
    X_PE.init(N_PE);
    Y_PE.init(N_PE);
    X_raw.init(N - N_PE);
    Y_raw.init(N - N_PE);

    // First N_PE -> PE
    for (int i = 0; i < N_PE; ++i) {
        X_PE(i) = alice_measurements(i);
        Y_PE(i) = bob_measurements(i);
    }

    for (int i = N_PE; i < N; ++i) {
        const int j = i - N_PE;
        X_raw(j) = alice_measurements(
            i); // Unnormalised key of Alice to be used for post-processing
        Y_raw(j) = bob_measurements(
            i); // Unnormalised key of Bob to be used for post-processing
    }

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL: alice_measurements = " << alice_measurements
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL: bob_measurements = " << bob_measurements
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL: X_PE = " << X_PE << std::endl;
    std::cerr << "CV_QKDPROTOCOL: Y_PE = " << Y_PE << std::endl;
    std::cerr << "CV_QKDPROTOCOL: X_raw = " << X_raw << std::endl;
    std::cerr << "CV_QKDPROTOCOL: Y_raw = " << Y_raw << std::endl;
#endif

    return {X_PE, Y_PE, X_raw, Y_raw};
}

std::tuple<double, double, double>
cvqkd_protocol::parameter_estimation_optical_fiber(
    const libbase::vector<double>& X_PE, const libbase::vector<double>& Y_PE)
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

    if (Epsilon_hat < 0) {
        Epsilon_hat = 0; // Epsilon_hat cannot be negative.
    }

    // Calculating estimate of transmittance: T_hat (eq. (5))
    // Note: I still need to add, v_el, N_0 and det_ff as serialized parameters
    // to the cv-qkd protocol for parameter estimation.
    T_hat = (t_hat * t_hat / detector_efficiency);

    // Calculating estimate for x_total_hat
    chi_total_hat = ((sigma2_hat) / (t_hat * t_hat)) - 1;

    return {T_hat, Epsilon_hat, chi_total_hat};
}

// Mutual Information for the GG02 protocol
double
cvqkd_protocol::calculate_mutual_information(double chi_total_hat)
{
    double V = m_modulation_variance + 1;

    // Equation to calculate I_AB for homodyne detection and under collective
    // attacks. It calculates the channel capacity of the quantum channel.

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
cvqkd_protocol::calculate_holevo_bound(double T_hat,
                                       double Epsilon_hat,
                                       double X_total_hat)
{
    double V = m_modulation_variance + 1;
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
    chi_BE = bosonic_entropy_G((lambda1 - 1.0) / 2.0) +
             bosonic_entropy_G((lambda2 - 1.0) / 2.0) -
             bosonic_entropy_G((lambda3 - 1.0) / 2.0) -
             bosonic_entropy_G((lambda4 - 1.0) / 2.0);

    // chi_BE_kbps = IBE * repetition_rate;
    return chi_BE; // In bits/pulse.
}

double
cvqkd_protocol::compute_beta_mdr(double code_rate)
{
    // beta is the reconciliation efficiency for MDR
    assert(code_rate >= 0.0);
    const double C = calculate_shannon_capacity_awgn();
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

    n_samples = get_codec_output_bits_n();

    assert(n_samples > 0);
    assert(smoothing_parameter > 0);

    // Equation (32) from Reference 2
    double delta_n =
        7 * std::sqrt(std::log2(2 / smoothing_parameter) / n_samples);

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  delta(n) = " << delta_n << std::endl;
#endif

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

    // Calculating N_PE: the number of samples used for parameter estimation.
    // N_PE = N (number of generated states) - n (size of codeword of the
    // codec)
    N_PE = alice_measurements.size() - get_codec_output_bits_n();

#if DEBUG >= 1
    std::cerr
        << "CV_QKDPROTOCOL: Number of states used for Parameter Estimation = "
        << N_PE << std::endl;
#endif

    // Perform split for parameter estimation.
    auto [X_PE, Y_PE, X_raw, Y_raw] =
        split(alice_measurements, bob_measurements);

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  Size of X_PE and Y_PE = " << X_PE.size()
              << "\t" << Y_PE.size() << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  Size of X_raw and Y_raw = " << X_raw.size()
              << "\t" << Y_raw.size() << std::endl;
#endif

    // Calculate parameter estimation using optical fiber.
    auto [T_hat, Epsilon_hat, chi_total_hat] =
        parameter_estimation_optical_fiber(X_PE, Y_PE);

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  T_hat = " << T_hat << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  Epsilon_hat = " << Epsilon_hat << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  chi_total_hat = " << chi_total_hat
              << std::endl;
#endif

// Gets modulation V_A after initialise method is called in qkd commsys.h
#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  modulation variance V_A = "
              << m_modulation_variance << std::endl;
#endif

    // Calculate Mutual Information I_AB
    I_AB = calculate_mutual_information(chi_total_hat);

    // Calculate Holevo Bound Chi_BE
    chi_BE = calculate_holevo_bound(T_hat, Epsilon_hat, chi_total_hat);

    // chi_be can never be negative
    if (chi_BE < 0) {
        chi_BE = 0;
    }

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  Mutual Information I_AB = " << I_AB
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  Holevo Bound Chi_BE = " << chi_BE
              << std::endl;
#endif

    // ------------ to delETE
    //     /* TODO: TO delete lines 236 and 237. I am just doing this for
    //     debugging
    //         * purposes since the framesize I started with was small/ */
    I_AB = 1.05;
    chi_BE = 0.82;

    // #if DEBUG >= 1
    //     std::cerr << "CV_QKDPROTOCOL:  Mutual Information I_AB = " << I_AB <<
    //     std::endl; std::cerr << "CV_QKDPROTOCOL:  Holevo Bound Chi_BE = " <<
    //     chi_BE << std::endl;
    // #endif
    // ----------- TO DELETE

    /* Checks whether the protocol is aborted or not to continue with the
     * Information Reconciliation stage. */

    if (I_AB > chi_BE) {
        MI_Check = true;

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL:  Mutual Information Check MI_Check = "
                  << MI_Check << std::endl;
#endif

        /* Gets variance VN from Bob's Gaussian Quantum Channel*/
        libbase::vector<double> bobs_channel_parameters;
        bobs_channel_parameters.init(1);
        bobs_channel_parameters = this->m_bob_channel->get_parameters();

        // CLI parameter of the gaussian quantum channel.
        SNR_linear =
            (this->m_modulation_variance) / (bobs_channel_parameters(0));

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL:  Variance VN = "
                  << (bobs_channel_parameters(0)) << std::endl;
        std::cerr << "CV_QKDPROTOCOL:  SNR_linear = " << SNR_linear
                  << std::endl;
#endif

        // Initialise and generate Bob's vector s which has size k.
        bob_vector_s.init(get_codec_input_bits_k());
        for (int i = 0; i < get_codec_input_bits_k(); ++i) {
            bob_vector_s(i) = (rng.ival(2) != 0);
        }

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: alice X_raw measurements = " << X_raw
                  << std::endl;
        std::cerr << "CV_QKDPROTOCOL: bob Y_raw measurements = " << Y_raw
                  << std::endl;
        std::cerr << "CV_QKDPROTOCOL: bob_vector_s = " << bob_vector_s
                  << std::endl;
#endif

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

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: bob_vector_c = " << bob_vector_c
                  << std::endl;
#endif

        /* Modulation step: Generate Vector M using BPSK modulation.*/

        // The direct_block_informed_embedder uses the embed method from the
        // base class block_informed_embedder.h.

        // Data to embed which is encoded bit vector C, converted from bool to
        // int.
        libbase::vector<int> data_to_embed(bob_vector_c.size());
        for (int i = 0; i < bob_vector_c.size(); ++i) {
            data_to_embed(i) = bob_vector_c(i);
        }

        // Vector M which stores the modulated signal.
        libbase::vector<double> vector_M;
        vector_M.init(get_codec_output_bits_n());

        embedder->set_blocksize(Y_raw.size()); // Y_raw are bob_measurements
                                               // after parameter estimation.

        embedder->embed(alphabet_size, data_to_embed, Y_raw, vector_M);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: vector_M = " << vector_M << std::endl;
#endif

        /* Demodulation step to get the Probability Table for the decoder. */

        // Instantiate the AWGN channel object.
        demodulation_channel = std::make_shared<libcomm::awgn1d>();

        // Convert SNR to dB
        double SNR_dB = 10.0 * std::log10(SNR_linear);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: demodulation_channel = "
                  << demodulation_channel->description() << std::endl;
        std::cerr << "CV_QKDPROTOCOL: SNR_linear = " << SNR_linear << std::endl;
        std::cerr << "CV_QKDPROTOCOL: SNR_dB = " << SNR_dB << std::endl;
#endif

        // Set SNR_db in AWGN channel
        demodulation_channel->set_parameter(SNR_dB);

        // Instantiate Probability Table.
        libbase::vector<libbase::vector<double>> prob_table;

        // Perform Demodulation to get Probability Table.
        embedder->extract(*demodulation_channel,
                          vector_M,
                          X_raw,
                          prob_table); // X_raw are alice_measurement after
                                       // parameter estimation.

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: prob_table = " << prob_table << std::endl;
#endif

        /*LDPC decoding using the prob_table to get vector s_hat */
        cdc->init_decoder(prob_table);

        auto decoded = libbase::vector<int>();
        cdc->decode(decoded);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: decoded = " << decoded << std::endl;
#endif

        // Copy decoded bits to vector s_hat  of Alice which is of bool type.
        libbase::vector<bool> vector_s_hat(get_codec_input_bits_k());

        for (int i = 0; i < decoded.size(); ++i) {
            vector_s_hat(i) = decoded(i);
        }

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: vector_s_hat = " << vector_s_hat
                  << std::endl;
        std::cerr << "CV_QKDPROTOCOL: hamming(vectors s, s_hat) = "
                  << libbase::hamming(bob_vector_s, vector_s_hat) << std::endl;
#endif

        /* Calculating Hashing for Vectors s and s_hat */
        std::uint32_t hash_hs = crc32_ieee<>::compute(bob_vector_s);
        std::uint32_t hash_hs_hat = crc32_ieee<>::compute(vector_s_hat);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: hash_hs = " << hash_hs << std::endl;
        std::cerr << "CV_QKDPROTOCOL: hash_hs_hat = " << hash_hs_hat
                  << std::endl;
#endif

        if (hash_hs == hash_hs_hat) {
            H_check = 1;
#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: H_check = " << H_check << std::endl;
#endif

            /* Calculate Beta for MDR: beta = R/C(S) taken from the Quasi Cyclic
             * Paper 2018, Mario Milicevic. C(S) is the Shannon Capacity of an
             * AWGN channel. */

            double R_code = cdc->rate();

#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: R_code = " << R_code << std::endl;
#endif

            double C_awgn_capacity = calculate_shannon_capacity_awgn();

            beta_mdr = R_code / C_awgn_capacity;

#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: C_awgn_capacity = " << C_awgn_capacity
                      << std::endl;
            std::cerr << "CV_QKDPROTOCOL: beta_mdr = " << beta_mdr << std::endl;
#endif

            /* Calculate length l of final secret key */
            len_secret_key = calculate_finite_size_effects_secret_key_length();

#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: len_secret_key = " << len_secret_key
                      << std::endl;
#endif

            // Sets length of secret keys KA and KB to later be able to retrieve
            // them for the results collector.
            libbase::vector<bool> final_secret_key_KA;
            libbase::vector<bool> final_secret_key_KB;

            final_secret_key_KA.init(len_secret_key);
            final_secret_key_KB.init(len_secret_key);

            if (len_secret_key == 0) {
                // Return empty keys.
                final_secret_key_KA.init(len_secret_key);
                final_secret_key_KB.init(len_secret_key);

                return {std::move(final_secret_key_KA),
                        std::move(final_secret_key_KB)};
            } else {
                // Continue and initialise Privacy Amplification system:
                // * alphabet size of 2
                // * length of final key after doing PA.
                // * length of pre-hashed key which in this case is the size of
                // vectors s and s_hat.
                pa_system.init(
                    len_secret_key, get_codec_input_bits_k(), alphabet_size);

                int starting_vector_len =
                    pa_system.generate_starting_vector_length();

#if DEBUG >= 1
                std::cerr << "CV_QKDPROTOCOL: PA system = "
                          << pa_system.description() << std::endl;
#endif

                // Generate starting vector.
                libbase::vector<bool> starting_vector =
                    pa_system.generate_starting_vector(starting_vector_len, 2);

                // Generate Standard Toeplitz matrix.
                libbase::matrix<bool> standard_toeplitz_matrix =
                    pa_system.generate_toeplitz_matrix(starting_vector);

                // Generates KB of Bob.
                final_secret_key_KB = pa_system.compute_hashed_key(
                    standard_toeplitz_matrix, bob_vector_s);

                // Generates KA of Alice.
                final_secret_key_KA = pa_system.compute_hashed_key(
                    standard_toeplitz_matrix, vector_s_hat);
            }
#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: final_secret_key_KA = "
                      << final_secret_key_KA << std::endl;
            std::cerr << "CV_QKDPROTOCOL: final_secret_key_KB = "
                      << final_secret_key_KB << std::endl;
#endif

            return {std::move(final_secret_key_KA),
                    std::move(final_secret_key_KB)};
        } else {

            H_check = 0;
#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: H_check = " << H_check << std::endl;
#endif

            libbase::vector<bool> final_secret_key_KA;
            libbase::vector<bool> final_secret_key_KB;

            len_secret_key = 0;

            // Sets length of secret keys KA and KB to zero to later be able to
            // retrieve it for the results collector.
            final_secret_key_KA.init(len_secret_key);
            final_secret_key_KB.init(len_secret_key);

            return {std::move(final_secret_key_KA),
                    std::move(final_secret_key_KB)};
        }
    } else {
        MI_Check = false;

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL:  Mutual Information Check MI_Check = "
                  << MI_Check << std::endl;
#endif

        len_secret_key = 0;

        libbase::vector<bool> final_secret_key_KA;
        libbase::vector<bool> final_secret_key_KB;

        final_secret_key_KA.init(len_secret_key);
        final_secret_key_KB.init(len_secret_key);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: final_secret_key_KA = "
                  << final_secret_key_KA << std::endl;
        std::cerr << "CV_QKDPROTOCOL: final_secret_key_KB = "
                  << final_secret_key_KB << std::endl;
#endif

        // print final keys
        return {std::move(final_secret_key_KA), std::move(final_secret_key_KA)};
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