#include "cvqkd_protocol.h"
#include "codec/ldpc.h"
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
cvqkd_protocol::init(
    qkd_commsys<gaussian_state, double, libbase::vector>* qkdcommsys)
{
    // Get the source generator from qkd_commsys
    std::shared_ptr<source<gaussian_state>> src_gen_base =
        qkdcommsys->get_src();
    assert(src_gen_base && "Commsys did not provide a source generator.");

    // Safely cast to the derived class we need
    auto& src_gen = dynamic_cast<quantum_gaussian_source&>(*src_gen_base);

    // Get modulation variance from the source
    m_modulation_variance = src_gen.get_VA();

    // Get Bob's quantum channel from qkd_commsys
    this->m_bob_channel = qkdcommsys->get_bob_channel();
    assert(this->m_bob_channel && "qkd_commsys did not provide Bob's channel.");
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

    libbase::vector<double> X_PE, Y_PE, X, Y;
    X_PE.init(N_PE);
    Y_PE.init(N_PE);
    X.init(N - N_PE);
    Y.init(N - N_PE);

    // First N_PE -> PE
    for (int i = 0; i < N_PE; ++i) {
        X_PE(i) = alice_measurements(i);
        Y_PE(i) = bob_measurements(i);
    }

    for (int i = N_PE; i < N; ++i) {
        const int j = i - N_PE;
        X(j) = alice_measurements(
            i); // Unnormalised key of Alice to be used for post-processing
        Y(j) = bob_measurements(
            i); // Unnormalised key of Bob to be used for post-processing
    }

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL: alice_measurements = " << alice_measurements
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL: bob_measurements = " << bob_measurements
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL: X_PE = " << X_PE << std::endl;
    std::cerr << "CV_QKDPROTOCOL: Y_PE = " << Y_PE << std::endl;
    std::cerr << "CV_QKDPROTOCOL: X = " << X << std::endl;
    std::cerr << "CV_QKDPROTOCOL: Y = " << Y << std::endl;
#endif

    return {X_PE, Y_PE, X, Y};
}

/**
 * @brief Estimates CV-QKD channel parameters using Covariance Matrix analysis.
 *
 * This function computes the statistical moments of the Alice (X) and Bob (Y) 
 * sequences to derive the channel characteristics. It assumes a linear channel 
 * model: Y = alpha * X + Z, where Z is the total noise (shot noise + excess noise).
 *
 * The estimation calculates:
 * 1. Variance of Alice's modulation (VA).
 * 2. Fading Channel coefficient (alpha), derived from Cov(X,Y).
 * 3. Total Noise Variance (VN), derived from Var(Y) and alpha.
 *
 * @note means_x and means_y are not 0 
 *
 * @param X_PE Input vector representing Alice's modulation values (Gaussian) used only for parameter estimation.
 * @param Y_PE Input vector representing Bob's measured values used only for parameter estimation.
 * - VA_hat: Estimated variance of Alice's signal.
 * - alpha_hat: Estimated channel gain/coupling coefficient.
 * - VN_hat: Estimated total noise variance.
 */
void
cvqkd_protocol::parameter_estimation(
    const libbase::vector<double>& X_PE, const libbase::vector<double>& Y_PE)
{
    /**
     * Reference for parameter estimation equations:
     * Equations of Covariance matrix were computed by Ryan Debono sent by email on 27/11/2025. 
     * - Recall that the covariance matrix is defined with the following format: 
     * | a c |
     * | c b |
     * where a = VA, b = (alpha)^2(VA) + VN, c = (alpha)(VA)
     * a is calculated from the variance of X_PE (Alice)
     * b is calculated from the variance of Y_PE (Bob)
     * c is calculated from the covariance of X_PE (Alice) and Y_PE (Bob) 
     */

    assert(X_PE.size() == Y_PE.size() && "X_PE and Y_PE must have same size.");

    const int m = X_PE.size();
    assert(m > 0 && "sample size m must be > 0.");

    // --- Step 1: Accumulate sums for Means and Moments ---
    long double sum_x = 0.0;
    long double sum_y = 0.0;
    long double sum_xx = 0.0;
    long double sum_yy = 0.0;
    long double sum_xy = 0.0;

    for (int i = 0; i < m; ++i) {
        double x = X_PE(i);
        double y = Y_PE(i);
        
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }

    // Calculate Means 
    double mean_x = sum_x / m; // to check if these gave to be zero 
    double mean_y = sum_y / m;

    // Calculate Variances (a, b) and Covariance (c) 
    // Using population variance (1/m) or sample variance (1/m-1). 
    // In QKD (large m), 1/m is standard.
    
    // a = Var(X) = E[X^2] - (E[X])^2
    double var_x = (sum_xx / m) - (mean_x * mean_x); 
    
    // b = Var(Y) = E[Y^2] - (E[Y])^2
    double var_y = (sum_yy / m) - (mean_y * mean_y);
    
    // c = Cov(X,Y) = E[XY] - E[X]E[Y]
    double cov_xy = (sum_xy / m) - (mean_x * mean_y);

    /* Map to Protocol Parameters to calculate VA_hat, alpha_hat and VN_hat */

    // a == VA_hat is calculated from variance of X_PE
    VA_hat = var_x;

    // c == (alpha_hat)(VA_hat) -> alpha_hat = c / VA_hat
    alpha_hat = 0.0;
    if (VA_hat > 1e-12) { // Protection against division by zero
        alpha_hat = cov_xy / VA_hat;
    }

    // b == (alpha_hat)^2(VA_hat) + VN_hat -> VN_hat = b - (alpha_hat^2 * VA_hat)
    VN_hat = var_y - (alpha_hat * alpha_hat * VA_hat);

    // Sanity check: Noise variance shouldn't be negative due to precision errors
    if (VN_hat < 0) VN_hat = 0.0;
}

/**
 * @brief Calculates the Mutual Information (I_AB) between Alice and Bob based on SNR only.
 *
 * It uses the Shannon-Hartley theorem adapted for the AWGN channel in the GG02 protocol.
 *
 * Mathematical Model:
 * I_AB = 0.5 * log2(1 + SNR)
 *
 * Where SNR is typically defined as: SNR_linear = (alpha)^2 (VA) / VN
 * - alpha: Variance of Alice's modulation.
 * - VA: modulation variance of Alice.
 * - VN: noise variance. 
 *
 * @references
 * [1] Villaseñor, Eduardo, et al. "Atmospheric effects on satellite-to-ground 
 * quantum key distribution using coherent states." GLOBECOM 2020.
 * [2] Ryan's equations. Definition of SNR is based on his equation
 * @param SNR in linear not in dB.  
 * @return double The mutual information in bits per pulse.
 */
double
cvqkd_protocol::calculate_mutual_information(double SNR_linear)
{
    /* References: 
    [1] Ryan's equations.
    [2] Villaseñor, Eduardo, et al. "Atmospheric effects on satellite-to-ground quantum key distribution using coherent states." 
    GLOBECOM 2020-2020 IEEE Global Communications Conference. IEEE, 2020.

    Where equation to calculate I_AB = 0.5 * log_2(1 + SNR)
    SNR is linear.  
    */

    I_AB = 0.5 * std::log2(1 + SNR_linear); // In bits/pulse
    // I_AB_kbps = I_AB * repetition_rate;

    return I_AB; 
}

/**
 * @brief Calculates the Holevo Bound (Chi_BE) using Ryan's derived equations.
 *
 * This function computes the maximum information available to Eve (Holevo quantity)
 * based on the symplectic eigenvalues of the covariance matrices.
 *
 * References:
 * [1] "Alternate Derivations.pdf", Section 3.1 Symplectic Eigenvalues and 3.2 Holevo Quantity.
 *
 * @param VA_hat The variance of Alice's modulation (a = sigma_X^2)[cite: eq. (48)].
 * @param alpha_hat The channel fading coefficient (alpha)[cite: eq. (24)].
 * @param VN_hat The noise variance (sigma_N^2)[cite: eq. (24)].
 * @return double The Holevo bound (bits/pulse).
 */
double
cvqkd_protocol::calculate_holevo_bound(double VA_hat,
                                       double alpha_hat,
                                       double VN_hat)
{
    // Ensure positive values for log calculations
    if (VA_hat <= 0 || VN_hat <= 0) return 0.0;

    // Map inputs to Covariance Matrix Parameters (Eq 37 - 40) ---
    // According to the PDF, the covariance parameters are derived as follows:
    // a = sigma_X^2 = VA
    const double a = VA_hat;
    
    // b = alpha^2 * sigma_X^2 + sigma_N^2
    const double b = (alpha_hat * alpha_hat * VA_hat) + VN_hat; 
    
    // c = alpha * sigma_X^2
    const double c = alpha_hat * VA_hat; 

    // Calculate Symplectic Eigenvalues lambda_1 and lambda_2 using equations (47) and (48)
    // z = sqrt((a + b)^2 - 4c^2)
    double term_z = std::sqrt(std::pow(a + b, 2) - 4.0 * c * c);

    // Lambda_1 (Eq 47, Eq 41)
    // Eq 47 is the expanded form of: 1/2 * (z + (b - a))
    const double lambda1 = 0.5 * (term_z + (b - a)); 

    // Lambda_2 (Eq 48, Eq 42)
    // Eq 48 is the expanded form of: 1/2 * (z - (b - a))
    const double lambda2 = 0.5 * (term_z - (b - a)); 

    // Lambda_3 (Eq 45)
    // Equation (45): lambda_3 = sqrt( a * (a - c^2/b) )
    assert(b > 1e-12 && "Division by zero: Variance b cannot be zero.");

    double term_inner = a - ((c * c) / b);

    // Sanity check: Inner term must be non-negative for sqrt
    if (term_inner < 0.0) term_inner = 0.0; 

    const double lambda3 = std::sqrt(a * term_inner);

    // Helper function to calculate g(x) (Von Neumann Entropy function) 
    // Based on Eq (52): g(x) = ((x+1)/2)*log2((x+1)/2) - ((x-1)/2)*log2((x-1)/2)
    auto calc_g = [](double x) -> double {
        double t1 = (x + 1.0) / 2.0;
        double t2 = (x - 1.0) / 2.0;
        
        double term1 = t1 * std::log2(t1); // x should also be positive 
        double term2 = (t2 > 0) ? (t2 * std::log2(t2)) : 0.0; // Handle limit x->1
        
        return term1 - term2; 
    };

#if DEBUG >= 1
    std::cout << "CV_QKDPROTOCOL:  lambda1 = " << lambda1
              << std::endl;
    std::cout << "CV_QKDPROTOCOL:  lambda2 = " << lambda2
              << std::endl;
    std::cout << "CV_QKDPROTOCOL:  lambda3 = " << lambda3 
              << std::endl;
#endif

    //  Calculate Holevo Quantity chi_BE (Eq 53)
    // chi = g(lambda1) + g(lambda2) - g(lambda3)
    double chi_BE = calc_g(lambda1) + calc_g(lambda2) - calc_g(lambda3); 
    return chi_BE; // In bits/pulse
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

    n_samples = cdc->output_block_size();

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

    libbase::vector<bool> final_secret_key_KA;
    libbase::vector<bool> final_secret_key_KB;

    // Set a default length of 0. This is updated upon success.
    len_secret_key = 0;

    // Calculating N_PE: the number of samples used for parameter estimation.
    // N_PE = N (number of generated states) - n (size of codeword of the
    // codec)
    assert(N_PE == alice_measurements.size() - cdc->output_block_size() && "N_PE does not match sifted key size minus n");


#if DEBUG >= 1
    std::cout 
        << "CV_QKDPROTOCOL: Number of states used for Parameter Estimation = "
        << N_PE << std::endl;
#endif

    // Perform split for parameter estimation.
    auto [X_PE, Y_PE, X, Y] =
        split(alice_measurements, bob_measurements);

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  Size of X_PE and Y_PE = " << X_PE.size()
              << "\t" << Y_PE.size() << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  Size of X and Y = " << X.size()
              << "\t" << Y.size() << std::endl << std::endl;
#endif

#if DEBUG >= 1
    std::cout << "CV_QKDPROTOCOL: Are VA_hat, VN_hat, alpha_hat calculated from parameter estimation?: " << estimate_parameters << std::endl;
#endif

    /*  Estimate Parameters 
        If estimate_parameters == true (default),
        VA_hat, VN_hat and alpha_hat are calculated from parameter estimation. 
        
        If estimate_parameters == false, 
        VA_hat, VN_hat and alpha_hat are taken directly from Bob's quantum channel. 
    */

    if(estimate_parameters)
    {
        // Calculate parameters from parameter estimation using Ryan's derived equations. 
        parameter_estimation(X_PE, Y_PE);

#if DEBUG >= 1
    std::cout << "Calculate Parameters from parameter estimation: " << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  VA_hat = " << VA_hat << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  alpha_hat = " << alpha_hat << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  VN_hat = " << VN_hat << std::endl << std::endl;
#endif
    }
    else
    {
        // Get the parameters directly from the objects.
        VA_hat = m_modulation_variance; 
        alpha_hat = m_bob_channel->get_alpha();
        VN_hat = m_bob_channel->get_VN();
        
#if DEBUG >= 1
    std::cout << "Parameters are taken directly from objects: " << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  VA_hat = " << VA_hat << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  alpha_hat = " << alpha_hat << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  VN_hat = " << VN_hat << std::endl << std::endl;
#endif
    }

    /* Gets variance VN from Bob's Gaussian Quantum Channel*/
    libbase::vector<double> bobs_channel_parameters;
    bobs_channel_parameters.init(1);
    bobs_channel_parameters = this->m_bob_channel->get_parameters();

    // Get alpha from the quantum gaussian channel of Bob
    alpha = this->m_bob_channel->get_alpha(); 

#if DEBUG >= 1
    std::cout << "Parameters directly from objects: " << std::endl;
    std::cout << "CV_QKDPROTOCOL:  modulation variance V_A = "
              << m_modulation_variance << std::endl;
    std::cout << "CV_QKDPROTOCOL:  Fading Coefficient alpha = " << alpha
              << std::endl;
#endif 

    // CLI parameter of the gaussian quantum channel.
    // TO CONFIRM whether I also need to serialize this in the 
    // the cvqkdprotocol.cpp as part of the switch as I did for DV-QKD.    
    SNR_linear =
        (alpha * alpha) * (this->m_modulation_variance) / (bobs_channel_parameters(0));

    // Convert SNR to dB
    double SNR_dB = 10.0 * std::log10(SNR_linear);

    // Calculate Mutual Information I_AB
    I_AB = calculate_mutual_information(SNR_linear);

    // Calculate Holevo Bound Chi_BE
    chi_BE = calculate_holevo_bound(VA_hat, alpha_hat, VN_hat);

    // chi_be can never be negative
    if (chi_BE < 0) {
        chi_BE = 0; // chi can never be negative.
    }

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL:  Mutual Information I_AB = " << I_AB
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL:  Holevo Bound Chi_BE = " << chi_BE
              << std::endl;
#endif 

    /* Checks whether the protocol is aborted or not to continue with the
     * Information Reconciliation stage. */
    MI_Check = (I_AB > chi_BE);

    if (MI_Check) {

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL:  Mutual Information Check MI_Check = true"
                  << std::endl;
#endif

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL:  Variance VN = "
                  << (bobs_channel_parameters(0)) << std::endl;
        std::cerr << "CV_QKDPROTOCOL:  SNR_linear = " << SNR_linear
                  << std::endl;
        std::cerr << "CV_QKDPROTOCOL:  SNR_dB = " << SNR_dB
                  << std::endl << std::endl;
#endif

        // Initialise and generate Bob's vector s which has size k.
        bob_vector_s.init(cdc->input_block_size());
        for (int i = 0; i < cdc->input_block_size(); ++i) {
            bob_vector_s(i) = (rng.ival(2) != 0);
        }

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: (Alice) X = " << X
                  << std::endl;
        std::cerr << "CV_QKDPROTOCOL: (Bob) Y  = " << Y
                  << std::endl;
        std::cerr << "CV_QKDPROTOCOL: bob_vector_s = " << bob_vector_s
                  << std::endl;
#endif

        // Generate Vector C of size n from Bob's vector s.
        libbase::vector<int> encoded_int(cdc->output_block_size());
        // Convert vector<bool> to vector<int>
        libbase::vector<int> bob_vector_int(bob_vector_s);

        // Encodes Vector S of Bob to get Vector C.
        cdc->encode(bob_vector_int, encoded_int);
        // Convert vector<int> to vector<bool>
        const libbase::vector<bool> bob_vector_c(encoded_int);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: bob_vector_c = " << bob_vector_c
                  << std::endl;
#endif

        /* Modulation step: Generate Vector M using BPSK modulation.*/

        // The direct_block_informed_embedder uses the embed method from the
        // base class block_informed_embedder.h.

        // Convert vector<bool> to vector<int>
        const libbase::vector<int> data_to_embed(bob_vector_c);

        // Vector M of size n which stores the modulated signal.
        libbase::vector<double> vector_M;
        vector_M.init(cdc->output_block_size());

        // Y are bobs measurements excluding those used for parameter estimation
        embedder->set_blocksize(Y.size()); 

        embedder->embed(alphabet_size, data_to_embed, Y, vector_M);

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: vector_M = " << vector_M << std::endl;
#endif

        /* Demodulation step to get the Probability Table for the decoder. */

        // Instantiate the AWGN channel object.
        demodulation_channel = std::make_shared<libcomm::awgn1d>();

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: demodulation_channel = "
                  << demodulation_channel->description() << std::endl;
#endif

        // Set SNR_db in AWGN channel
        demodulation_channel->set_parameter(SNR_dB);

        // Instantiate Probability Table.
        libbase::vector<libbase::vector<double>> prob_table;

        // Perform Demodulation to get Probability Table.
        embedder->extract(*demodulation_channel,
                          vector_M,
                          X,
                          prob_table); // X are alice_measurement after
                                       // parameter estimation.

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: prob_table = " << prob_table << std::endl;
#endif

        /*LDPC decoding using the prob_table to get vector s_hat */
        cdc->init_decoder(prob_table);

        auto decoded = libbase::vector<int>();
        cdc->decode_message(decoded);

        // Copy decoded bits to vector s_hat  of Alice which is of bool type.
        libbase::vector<bool> vector_s_hat(cdc->input_block_size());

        for (int i = 0; i < decoded.size(); ++i) {
            vector_s_hat(i) = decoded(i);
        }

#if DEBUG >= 1
        std::cerr << "CV_QKDPROTOCOL: (Alice) decoded vector_s_hat = " << vector_s_hat
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

        H_check = (hash_hs == hash_hs_hat);

        if (H_check) {
#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: H_check = true" << std::endl;
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

            if (len_secret_key > 0) {
                // Perform Privacy Amplification:
                // * alphabet size of 2
                // * length of final key after doing PA.
                // * length of pre-hashed key which in this case is the size of
                // vectors s and s_hat of size k.
                pa_system.init(
                    len_secret_key, cdc->input_block_size(), alphabet_size);

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
        } else {

#if DEBUG >= 1
            std::cerr << "CV_QKDPROTOCOL: H_check = false" << H_check
                      << std::endl;
#endif
        }
    } else {
#if DEBUG >= 1
        std::cerr
            << "CV_QKDPROTOCOL:  Mutual Information Check MI_Check = false"
            << std::endl;
#endif
    }

    // If any check failed, len_secret_key will still be 0.
    // Initialize empty keys in that case.
    if (len_secret_key == 0) {
        final_secret_key_KA.init(0);
        final_secret_key_KB.init(0);
    }

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL: len_secret_key = " << len_secret_key
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL: final_secret_key_KA = " << final_secret_key_KA
              << std::endl;
    std::cerr << "CV_QKDPROTOCOL: final_secret_key_KB = " << final_secret_key_KB
              << std::endl;
#endif

    // print final keys
    return {std::move(final_secret_key_KA), std::move(final_secret_key_KB)};
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
    // format version
    sout << "# Version" << std::endl;
    sout << 1 << std::endl;
    sout << "# N_PE" << std::endl; // # used for parameter estimation
    sout << N_PE << std::endl; 
    sout << "# VA, VN, alpha from parameter estimation?" << std::endl;
    sout << int(estimate_parameters) << std::endl;
    sout << "# Smoothing Parameter" << std::endl;  // Smoothing parameter bar epsilon is used to calculate the final
    // length of the secret key.
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

    // get format version
    int version;
    sin >> libbase::eatcomments >> version;
    sin >> libbase::eatcomments >> N_PE >> libbase::verify;
    sin >> libbase::eatcomments >> estimate_parameters >> libbase::verify;
    sin >> libbase::eatcomments >> smoothing_parameter >> libbase::verify;
    sin >> libbase::eatcomments >> alphabet_size >> libbase::verify;
    // we have to serialise this as a codec object, then do a dynamic conversion
    std::shared_ptr<codec<libbase::vector>> _cdc;
    sin >> libbase::eatcomments >> _cdc >> libbase::verify;
    cdc = std::dynamic_pointer_cast<codec_coset<libbase::vector>>(_cdc);
    assert(cdc);
    sin >> libbase::eatcomments >> embedder >> libbase::verify;

    // check that all assumptions hold
    assertalways(alphabet_size == 2); // we only support binary for now
    assertalways(cdc->num_inputs() == alphabet_size); // codec input has to be binary
    assertalways(cdc->num_outputs() == embedder->num_symbols()); // codec output has to match embedder input

    return sin;
}

const serializer cvqkd_protocol::shelper("qkd_protocol",
                                         "cvqkd_protocol",
                                         cvqkd_protocol::create);

} // namespace libcomm