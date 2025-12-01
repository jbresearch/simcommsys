#include "dvqkd_protocol.h"
#include <sstream>
#include <iostream>
#include <cmath>    
#include <limits>  

using libbase::serializer;


namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 2
#endif

// Returns description of the protocol
std::string
dvqkd_protocol::description() const
{
    return "BB894 protocol with single polarization";
}

void
dvqkd_protocol::init(qkd_commsys<qubit, bool, libbase::vector>* qkdcommsys)
{
    // // Get the source generator from commsys
    // std::shared_ptr<source<gaussian_state>> src_gen_base = qkdcommsys->get_src();
    // assert(src_gen_base && "Commsys did not provide a source generator.");

    // // Safely cast to the derived class we need
    // auto& src_gen = dynamic_cast<quantum_bb84_source&>(*src_gen_base);

    // Get Bob's quantum channel from qkd_commsys
    this->m_bob_channel = qkdcommsys->get_bob_channel();
    assert(this->m_bob_channel && "qkd_commsys did not provide Bob's channel.");
}

void
dvqkd_protocol::set_source_sequence(const libbase::vector<qubit>& source_sequence)
{
    this->m_source_sequence = &source_sequence;
}

std::pair<bool, bool>
dvqkd_protocol::get_alice_choice_from_qubit(const libcomm::qubit& q)
{
    // Define the values Alice uses
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double epsilon = 1e-9; // A small tolerance for float comparison

    // Get the internal amplitudes.
    std::complex<double> alpha = q.get_comp_basis_0();
    std::complex<double> beta = q.get_comp_basis_1();

    // Based on the quantum_bb84_source.h, only the real parts need to be checked.

    // Case 1: State |0> (bit=0, basis=0)
    // alpha=1.0, beta=0.0
    if (std::abs(alpha.real() - 1.0) < epsilon && std::abs(beta.real()) < epsilon) {
        return {false, false}; // bit=0, basis=0 (Z)
    }

    // Case 2: State |1> (bit=1, basis=0)
    // alpha=0.0, beta=1.0
    if (std::abs(alpha.real()) < epsilon && std::abs(beta.real() - 1.0) < epsilon) {
        return {true, false}; // bit=1, basis=0 (Z)
    }

    // Case 3: State |+> (bit=0, basis=1)
    // alpha=1/sqrt(2), beta=1/sqrt(2)
    if (std::abs(alpha.real() - inv_sqrt2) < epsilon && std::abs(beta.real() - inv_sqrt2) < epsilon) {
        return {false, true}; // bit=0, basis=1 (X)
    }

    // Case 4: State |-> (bit=1, basis=1)
    // alpha=1/sqrt(2), beta=-1/sqrt(2)
    if (std::abs(alpha.real() - inv_sqrt2) < epsilon && std::abs(beta.real() + inv_sqrt2) < epsilon) {
        return {true, true}; // bit=1, basis=1 (X)
    }

    // If it's none of these, it's a state we don't recognize (e.g., noisy)
    throw std::runtime_error("Unknown qubit state. Not a valid Alice state.");
}


// Note: here I replaced libbase::vector with the std::vector only for the
    // observables. Returns the observables of Bob
std::vector<std::unique_ptr<observable<bool>>>
dvqkd_protocol::get_bob_observables(int framesize)
{
    std::vector<std::unique_ptr<observable<bool>>> observables;
    observables.reserve(framesize);

    bob_basis_vector.init(framesize);

    for (int i = 0; i < framesize; ++i) {
        if (rng.ival(2) == 0) {
            observables.push_back(
                std::make_unique<computational_observable>());
            bob_basis_vector(i) = 0;
        } else {
            observables.push_back(std::make_unique<hadamard_observable>());
            bob_basis_vector(i) = 1;
        }
    }

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Basis of Bob: " << bob_basis_vector << std::endl;
    // std::cout << "DV_QKDPROTOCOL: If b'(i) = 0 it is a Computational observable otherwise it is a Hadamard observable. "
                // << std::endl;
#endif

    std::cout << bob_basis_vector << std::endl;
    return observables;
}


// Returns the observables of Alice
std::vector<std::unique_ptr<observable<bool>>>
dvqkd_protocol::get_alice_observables(int framesize)
{
    std::vector<std::unique_ptr<observable<bool>>> observables;
    observables.reserve(framesize);

    alice_basis_vector.init(framesize); // vector b (basis)
    alice_bit_vector.init(framesize);   // vector a (bit)

    /* Ensure the source sequence has been set by set_source_sequence().
    set_source_sequence() is called in the fullcycle method found in
    qkd_commsys.cpp
    */
    assert(m_source_sequence &&
            "Source sequence was not set in dvqkd_protocol");
    assert(m_source_sequence->size() == framesize &&
            "Source sequence size mismatch");

    for (int i = 0; i < framesize; ++i) {

        // Get the qubit from the stored sequence
        const qubit& q = (*m_source_sequence)(i);

        // Reverse-engineer the bit and basis from the qubit state
        std::pair<bool, bool> alice_choice = get_alice_choice_from_qubit(q);

        bool bit = alice_choice.first;
        bool basis = alice_choice.second;

        // Store them in the protocol's member vectors
        alice_bit_vector(i) = bit;     // This is Alice's bit vector a
        alice_basis_vector(i) = basis; // This is Alice's basis vector b

        // Create the corresponding fake observable for Alice
        if (basis == 0) { // Z-basis (Computational)
            observables.push_back(
                std::make_unique<fake_computational_observable>());
        } else { // X-basis (Hadamard)
            observables.push_back(
                std::make_unique<fake_hadamard_observable>());
        }
    }


#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Basis of Alice: " << alice_basis_vector << std::endl;
    // std::cout << "DV_QKDPROTOCOL: If b(i) = 0 it is a fake Computational observable otherwise it is a fake Hadamard observable. "
    //             << std::endl;
#endif

        return observables;
    }

/*! \brief Performs split for parameter estimation 
 *
 * @param X_PE  A bool vector which holds the measurement values for PE for Alice 
 * @param X Alice's key. It holds the remaining measurement values after sifting excluding PE. 
 * @param Y_PE  A bool vector which holds the measurement values for PE for Bob 
 * @param Y Bob's key with errros. It holds the remaining measurement values after sifting excluding PE. 
 */
void
dvqkd_protocol::split(libbase::vector<bool>& alice_measurements,
                      libbase::vector<bool>& bob_measurements)
{
    assert(alice_measurements.size() == bob_measurements.size() &&
           "Alice and Bob's measurement vector sizes are not equal.");

    assert(N_PE > 0 && "N_PE must be > 0.");

    const int N = alice_measurements.size();
    assert(N==bob_measurements.size());

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
}

/*! \brief Method to calculate the binary entropy function */
double dvqkd_protocol::binary_entropy(double p) {
    // Probability must be between 0 and 1
    if (p < 0.0 || p > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Handle Edge Cases: 0 and 1
    // log2(0) is -inf, which results in NaN when multiplied by 0.
    // Mathematically, the limit of p*log(p) as p->0 is 0.
    if (p == 0.0 || p == 1.0) {
        return 0.0;
    }

    // Calculate Entropy (in bits)
    return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
}

/*! \brief Calculates the Finite-Key Secure Length (Equation 2 from Tomamichel et al., 2012)
 *
 * @param n_d        Length of the keys after performing the split. This is the length of vectors X and Y after PE. 
 * @param k_d        Length of the parameter estimation bits. 
 * @param Q_tol      The maximum tolerated QBER (typically around 7%)
 * @param leak_EC    Bits revealed during error correction (syndrome length)
 * @param eps_sec    Security parameter (e.g., 1e-10). Smoothing parameter that ensures composable security. 
 * @param eps_cor    Correctness parameter (e.g., 1e-15). Likelihood that even after EC, the two keys still differ.
 * @param q          Source quality factor (default 1.0 for perfect qubits)
 * @return           Final secure key length 'l' (floored to 0 if negative)
 */

const int dvqkd_protocol::calculate_finite_size_effects_secret_key_length() 
{
     /*
    References for the equation to claculate the length of the secret key:
    1. Eq (5.108), Ramona Wolf, "Quantum Key Distribution..."
    2. Eq (2), Tomamichel et al., Nature Comms 3.1 (2012)
    */

    #if DEBUG >= 1
       std::cout <<  "DV_QKDPROTOCOL: eps_cor = " << eps_cor << std::endl; 
        std::cout << "DV_QKDPROTOCOL: eps_sec = " << eps_sec << std::endl;
    #endif

    double n_d = static_cast<double>(X.size()); // excludes bits used for PE.
    double k_d = static_cast<double>(N_PE);
    int q = 1; 
    int leak_EC = get_codec_output_bits_n() - get_codec_input_bits_k(); // size of syndrome

    #if DEBUG >= 1
       std::cout << "DV_QKDPROTOCOL: Length of key after split and PE n_d = " << n_d << std::endl; 
       std::cout << "DV_QKDPROTOCOL: Length of PE bits k_d = " << k_d << std::endl;
       std::cout << "DV_QKDPROTOCOL: Leak_EC = " << leak_EC << std::endl;
    #endif

    /* Calculate statistical fluctuation term 'mu'
    Formula: mu = sqrt( ((n+k)/(n*k)) * ((k+1)/k) * ln(2/eps_sec) )
    mu refers to the statistical correction factor; when one calculates the QBER for P.E.
    since one uses a finite size for X_PE and Y_PE, the mu is that margin of error.
    */

    double term1 = (n_d + k_d) / (n_d * k_d);
    double term2 = (k_d + 1.0) / k_d;
    double term3 = std::log(2.0 / eps_sec);
    
    double mu = std::sqrt(term1 * term2 * term3);

    // Calculate the "worst-case" error rate
    double Q_tol = 0.07; // Assuming Q_tol is 7% which is tighter than the 10%. 
    double Q_worst_case = Q_tol + mu;

    /* QBER is calculated in the parameter estimation method. 
    // Checks that it is not >= the maximum tolerable qber. */  
    if (QBER >= Q_worst_case) {
        return 0.0; // Returns a final length of 0
    }

    #if DEBUG >= 1
       std::cout << "DV_QKDPROTOCOL: mu  = " << mu << std::endl; 
       std::cout << "DV_QKDPROTOCOL: Q_worst_case = " << k_d << std::endl;
    #endif

    // Calculate the correction term (Delta)
    // Formula: log2( 2 / (eps_cor * eps_sec^2) )
    double delta = std::log2(2.0 / (eps_cor * std::pow(eps_sec, 2)));

    // Calculate final length 'l'
    // Formula: l = n * [ q - h(Q_tol + mu) ] - leak_EC - delta
    double privacy_amplification_term = binary_entropy(Q_worst_case);

    #if DEBUG >= 1
       std::cout << "DV_QKDPROTOCOL: Result of Binary Entropy fn = " << privacy_amplification_term << std::endl; 
    #endif
    
    /* Note: 
    Finite-key analysis requires n and k to be in the order of 10^4 to 10^5 to produce a positive key length. 
    With single-digit inputs, the uncertainty is too high to guarantee any secrecy.
    */

    double l = n_d * (q - privacy_amplification_term) - static_cast<double>(leak_EC) - delta;

    // Return 0 if the result is negative
    return static_cast<int>(std::floor(std::max(0.0, l))); 
}

/*! \brief Estimates the channel error rate (QBER) and calculates the final secure key length.
 *
 * This method compares the subsets of bits reserved for Parameter Estimation (PE)
 * from Alice and Bob to calculate the Quantum Bit Error Rate (QBER). Based on this
 * QBER and finite-size security bounds, it computes the available length for the
 * final secret key.
 *
 * @param X_PE Alice's bit vector reserved for parameter estimation.
 * @param Y_PE Bob's bit vector reserved for parameter estimation.
 */
void
dvqkd_protocol::parameter_estimation(
    const libbase::vector<bool>& X_PE, const libbase::vector<bool>& Y_PE)
{
    /*
    References to calculate the QBER:
    Reference 1 of Equation used is pg. 110 from the book of Ramona Wolf.
    Book is titled "Quantum Key Distribution: An Introduction With Exercises"
    Equation Number: (4.29)

    Reference 2 for QBER equation:
    Box 1: Protocol Definition under section Parameter Estimation from the paper titled:
    "Tomamichel, Marco, et al. "Tight finite-key analysis for quantum cryptography." Nature communications 3.1 (2012): 634."
    */

    // Calculate the QBER between vectors X_PE of Alice and Y_PE of Bob
    QBER = (1.0/static_cast<double> (X_PE.size())) * libbase::hamming(X_PE,Y_PE);

    // Calculate the final length of the secret l with finite size effects
    // len_secret_key = calculate_secure_key_length(X_raw.size(), X_PE.size(), Q_tol, syndrome_size);
    // len_secret_key = calculate_secure_key_length(100000, 50000, Q_tol, 50000); // Answer l = 4045 
#if DEBUG >= 1
    std::cout << "Calculating the secret key length:"  << std::endl;
#endif

    len_secret_key = calculate_finite_size_effects_secret_key_length(); 
    
#if DEBUG >= 2
    std::cout << "DV_QKDPROTOCOL: Parameter Estimation Calculations" 
              << std::endl;
    std::cout << "DV_QKDPROTOCOL: QBER = " << QBER
              << std::endl;
    std::cout << "DV_QKDPROTOCOL: len_secret_key = " << len_secret_key
              << std::endl;
#endif
} 

/*!
 * \brief Packs a stream of raw bits into integer symbols.
 *
 * This function is used to bridge the gap between a binary QKD key and a 
 * Non-Binary Codec (e.g., GF(16)). It groups 'm' consecutive bits into 
 * a single integer symbol.
 *
 * \param[in] bits  The raw boolean vector from the QKD sifting process (e.g., X_raw).
 * \param[in] m     The number of bits per symbol (e.g., 4 for GF(16)). 
 * If m <= 1, the function acts as a simple cast from bool to int.
 *
 * \return A vector of integers where each element represents a symbol 
 * formed by 'm' bits. The size will be floor(bits.size() / m).
 *
 * \note This function assumes MSB-first packing (Big Endian). 
 * Example (m=4): Bits [1, 0, 0, 1] becomes Integer 9.
 */
libbase::vector<int> 
dvqkd_protocol::pack_bits_to_symbols(const libbase::vector<bool>& bits, int m) 
{
    // Safety check: if m is less than 1 (e.g., binary), treat as 1
    if (m < 1) m = 1;

    /* Calculate number of symbols
    Any "leftover" bits at the end of the stream that don't fill a full symbol are dropped.*/
    int num_symbols = bits.size() / m;
    
    libbase::vector<int> symbols;
    symbols.init(num_symbols);

    for (int i = 0; i < num_symbols; ++i) { 
        int value = 0;
        for (int b = 0; b < m; ++b) {
            /* Pack MSB first: The first bit in the chunk goes to the highest position.
            Example for m=4: 
             b=0 (1st bit) -> shifted left by 3
             b=3 (4th bit) -> shifted left by 0
            */
            if (bits(i * m + b)) {
                // Shift 1 bit to the left start with the MSB and XOR with the value 
                value |= (1 << (m - 1 - b));
            }
        }
        symbols(i) = value;
    }

    return symbols;
}

// Returns final secret keys KA and KB.
std::pair<libbase::vector<bool>, libbase::vector<bool>>
dvqkd_protocol::postprocess(libbase::vector<bool>&& alice_measurements,
                            libbase::vector<bool>&& bob_measurements)
{

    libbase::vector<bool> final_secret_key_KA;
    libbase::vector<bool> final_secret_key_KB;

    // Set a default length of 0. This is updated upon success.
    len_secret_key = 0;

    /* Sifting Step */
    // In this step we need to discard the bits where the basis vectors of Alice (vector b) and Bob (vector b') do not match.
    // Get and store the indices of the elements of the basis vectors that won't match.

    std::vector<int> diff_indices;

    /* alice_basis_vector is assigned in get_alice_observables method.
     Similarly for Bob, bob_basis_vector is assigned in get_bob_observables method.  
    */
    if (bob_basis_vector.size() != alice_basis_vector.size()) {
        std::cerr << "Vectors are not the same size, so all indices beyond the smaller size will be different." << std::endl;
    } else {
        for (int i = 0; i < alice_basis_vector.size(); ++i) {
            // Get and store the indices of the elements of the basis vectors that won't match.
            if (alice_basis_vector(i) != bob_basis_vector(i)) {
                diff_indices.push_back(i);
            }
        }
    }

#if DEBUG >= 1
        std::cerr << "DV_QKDPROTOCOL: **** SIFTING PROCESS **** " << std::endl;
        std::cerr << "DV_QKDPROTOCOL: Indices of Mismatch in the elements of the basis vectors = " << diff_indices
                  << std::endl;
#endif

    if (alice_measurements.size() != bob_measurements.size()) {
        std::cerr << "Measurement vectors are not the same size." << std::endl;
    }

    // Delete the bits from both measurement vectors with those indices.
    const int n_original = alice_measurements.size();
    const int n_sifted = n_original - (int)diff_indices.size();

    libbase::vector<bool> sifted_alice_key;
    libbase::vector<bool> sifted_bob_key;
    sifted_alice_key.init(n_sifted);
    sifted_bob_key.init(n_sifted);

    int sifted_idx = 0;
    int diff_idx = 0;

    for (int i = 0; i < n_original; ++i) {
        // Check if the current index i is the next one to exclude.
        if (diff_idx < (int)diff_indices.size() && diff_indices[diff_idx] == i) {
            // Exclude index. Move to the next index in diff_indices.
            diff_idx++;
        } else {
            // If the index's basis matched, keep the measurement.
            sifted_alice_key(sifted_idx) = alice_measurements(i);
            sifted_bob_key(sifted_idx) = bob_measurements(i);
            sifted_idx++;
        }
    }

    assert(sifted_idx == n_sifted);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Original measurement vector size = " << n_original << std::endl;
    std::cout << "DV_QKDPROTOCOL: Size of sifted keys = " << n_sifted << std::endl;
    std::cout << "DV_QKDPROTOCOL: Sifted Alice Key = " << sifted_alice_key << std::endl;
    std::cout << "DV_QKDPROTOCOL: Sifted Bob Key = " << sifted_bob_key << std::endl;
#endif

    /* Calculating N_PE: the number of samples used for parameter estimation.
    N_PE = Size of sifted key - n (size of codeword of the codec) */
    N_PE = sifted_alice_key.size() - get_codec_output_bits_n(); 

    // Perform split for parameter estimation.
    split(sifted_alice_key, sifted_bob_key);

#if DEBUG >= 1
        std::cout << "---- Perform Split -----" << std::endl;
        std::cout << "DV_QKDPROTOCOL: size of N_PE = " << N_PE << std::endl;
        std::cout << "DV_QKDPROTOCOL: Y_PE = "
                  << Y_PE << std::endl;
        std::cout << "DV_QKDPROTOCOL: size of Y_PE = "
                  << Y_PE.size() << std::endl; // to delete
        std::cout << "DV_QKDPROTOCOL: Y = "
                  << Y << std::endl;
        std::cout << "DV_QKDPROTOCOL: size of Y = "
                  << Y.size() << std::endl; // to delete
        std::cout << "DV_QKDPROTOCOL: X_PE = "
                  << X_PE << std::endl;
        std::cout << "DV_QKDPROTOCOL: size of X_PE = "
                  << X_PE.size() << std::endl; // to delete
        std::cout << "DV_QKDPROTOCOL: X = "
                  << X << std::endl;
        std::cout << "DV_QKDPROTOCOL: size of X = "
                  << X.size() << std::endl; // to delete

#endif

    // Aborts of sizes of sifted keys are not equal
    assert(sifted_alice_key.size() == sifted_bob_key.size());
    
    // Perform Parameter Estimation to calculate the QBER and l 
    parameter_estimation(X_PE, Y_PE); 

#if DEBUG >= 1
        std::cout << "DV_QKDPROTOCOL: Estimated QBER = " << QBER << std::endl;
        std::cout << "DV_QKDPROTOCOL: Secret Key Length l = " << len_secret_key << std::endl;
#endif

    /* (Alice) (Inverse Mapping) 
    Convert X_raw to binary or non-binary to be able to calculate the syndrome
    Convert libbase::vector<bool> -> libbase::vector<int> */

    // Get the alphabet size from the loaded codec (e.g., 2 for GF2, 16 for GF16)
    int q = cdc->num_outputs(); 
    
    // Calculate log2(q) to get m (bits per symbol)
    // Examples: q=2 -> m=1, q=4 -> m=2, q=16 -> m=4
    int m = 0;
    if (q > 0) {
        int temp = q;
        while (temp >>= 1) m++;
    }

    // Safety fallback for binary or uninitialized codec
    if (m == 0) m = 1; 

    // Alice's side convert bits (bool) -> symbols (int)
    libbase::vector<int> X_int = pack_bits_to_symbols(X, m);

    //(Alice) Calculate the syndrome of X of size (n-k)
    libbase::vector<int> calculated_syndrome(cdc->output_block_size() - cdc->input_block_size());
    cdc->calculate_syndrome(X_int, calculated_syndrome); 

#if DEBUG >= 1
        std::cout << "DV_QKDPROTOCOL: Alice's Inverse Mapped vector = " << X_int << std::endl;
        std::cout << "DV_QKDPROTOCOL: Alice's Calculated Syndrome = " << calculated_syndrome << std::endl;
#endif

    // (Bob) Convert vector Y from libbase::vector<int> to libbase::vector<gf2>
    // Convert vector<int> to vector<bool>
    const libbase::vector<libbase::gf2> Y_gf2(Y);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Y converted to GF2 "<< std::endl;
    print_gf_vector_as_ints(Y_gf2);
#endif

    // Set Ps of the qsc channel which is the estimated QBER from PE
    /******* TO REMOVE QBER2 ---this is just a hack because framesize of 
     * N_PE is too small for now, so the QBER is being skewed. 
     */
    double QBER2 = 1.0/7.0; // Actual error was 0.17.

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: QBER2 (has to be deleted) = " << QBER2 << std::endl;
#endif

    // demodulation_channel->set_parameter(QBER);
    demodulation_channel->set_parameter(QBER2); // To delete and revert to QBER

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Description of Demodulation Channel" << demodulation_channel->description() << std::endl;
#endif

    // Initialise probability table.
    auto prob_table = libbase::vector<libbase::vector<double>>(Y_gf2.size());

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Modem Description: " << mdm->description() << std::endl;
#endif

    // Setting block size of modem
    mdm->set_blocksize(libbase::size_type<libbase::vector>(Y_gf2.size()));

    /* (Bob) Demodulate the received codeword to get the required probability table
    Format of Table: P(bit 0), P(bit 1).
    Demodulation channel is a qsc channel which is initialised in serialize sin
    This was required due to RNG intialisation in the seedfrom fn found in dvqkd_protocol.h
    */
    mdm->demodulate(*demodulation_channel, Y_gf2, prob_table);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Probability Table: "
                  << prob_table << std::endl;
#endif

    /* (Bob) Inverse Map. 
    For binary this is a "map_straight"
    For converting from binary to non-binary this is a "map_dividing"
    */
    auto prob_table_encoded = libbase::vector<libbase::vector<double>>();
    map->inverse(prob_table, prob_table_encoded);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Probability Encoded obtained from Inverse Mapping: "
                  << prob_table_encoded << std::endl;
#endif

    auto decoded_bob_k_message = libbase::vector<int>(cdc->input_block_size());
    // (Bob) Perform Syndrome Decoding
    cdc->init_decoder(prob_table_encoded, calculated_syndrome);
    cdc->decode(decoded_bob_k_message);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Bob's Decoded message k: "
              << decoded_bob_k_message << std::endl;
#endif

    // Encode decoded Bob's k decoded message to get Y_hat to then compare to X
    libbase::vector<int> Y_hat_int(Y_gf2.size());
    cdc->encode(decoded_bob_k_message, Y_hat_int);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Bob's Y_hat_int: "
              << Y_hat_int << std::endl;
#endif

    // To double check with Johann on this:
    // TO VERIFY YOU HAVE TO ALSO DECODE X AND ENCODE IT AGAIN? 

    /******  VERIFICATION (to delete) ******/ 
    // Modulate codeword
    libbase::vector<libbase::gf2> modulated_codeword_p1(X_int.size());
    mdm->modulate(alphabet_size, X_int, modulated_codeword_p1); 

    // Transmit codeword through a QSC channel with Ps = 0.0
    double Ps_no_error = 0.0; 
    demodulation_channel->set_parameter(Ps_no_error);

    libbase::vector<libbase::gf2> received_codeword_p1(X_int.size());
    demodulation_channel->transmit(modulated_codeword_p1, received_codeword_p1);

    // Demodulate (Error free) 
    auto prob_table_p1_X_int = libbase::vector<libbase::vector<double>>(X_int.size());
    mdm->demodulate(*demodulation_channel, received_codeword_p1, prob_table_p1_X_int);

    auto decoded_alice_k_message = libbase::vector<int>(cdc->input_block_size());
    cdc->seedfrom(rng); 

    cdc->init_decoder(prob_table_p1_X_int, calculated_syndrome);
    cdc->decode(decoded_alice_k_message);

#if DEBUG >= 1
    std::cout << "*** DV_QKDPROTOCOL: Verifying decoded message ***" << std::endl;
    std::cout << "DV_QKDPROTOCOL: Decoded Alice message u of size k (no error): "
              << decoded_alice_k_message << std::endl;
#endif
    /******  END OF VERIFICATION ******/

    // Convert vector Y_hat_int to bool
    const libbase::vector<bool> Y_hat(Y_hat_int);

    /* Calculating Hashing for Vectors s and s_hat */
    std::uint32_t hash_X = crc32_ieee<>::compute(X);
    std::uint32_t hash_Y_hat = crc32_ieee<>::compute(Y_hat);

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: hash_X = " << hash_X << std::endl;
    std::cout << "DV_QKDPROTOCOL: hash_Y_hat = " << hash_Y_hat
                << std::endl;
#endif

    H_check = (hash_X == hash_Y_hat);

    if (H_check) {

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL: H_check = true" << std::endl;
#endif

    // length of secret key is calculated in parameter estimation step 
    final_secret_key_KA.init(len_secret_key);
    final_secret_key_KB.init(len_secret_key);

    // If l>0, continue with privacy amplification to get the final keys
    if (len_secret_key > 0) {
            /* Perform Privacy Amplification:
            param1: alphabet size of 2
            param2: length of final key after doing PA.
            param3: length of pre-hashed key which in this case is the size of
            vectors s and s_hat.
            */
            pa_system.init(
                len_secret_key, X.size(), alphabet_size);

            int starting_vector_len =
                pa_system.generate_starting_vector_length();

#if DEBUG >= 1
                std::cerr << "CV_QKDPROTOCOL: PA system = "
                          << pa_system.description() << std::endl;
#endif

            // Generate starting vector.
            libbase::vector<bool> starting_vector =
                pa_system.generate_starting_vector(starting_vector_len, alphabet_size);

            // Generate Standard Toeplitz matrix.
            libbase::matrix<bool> standard_toeplitz_matrix =
                pa_system.generate_toeplitz_matrix(starting_vector);

            // Generates KB of Bob.
            final_secret_key_KB = pa_system.compute_hashed_key(
                standard_toeplitz_matrix, Y_hat);

            // Generates KA of Alice.
            final_secret_key_KA = pa_system.compute_hashed_key(
                standard_toeplitz_matrix, X);
                        }
    }
    else
    {

#if DEBUG >= 1
    std::cerr << "CV_QKDPROTOCOL: H_check = false" << std::endl;
#endif

    len_secret_key = 0; // Return null as final secret keys
    final_secret_key_KA.init(len_secret_key);
    final_secret_key_KB.init(len_secret_key);
    }

    return {std::move(final_secret_key_KA), std::move(final_secret_key_KB)};
}

//! Serialize protocol
std::ostream&
dvqkd_protocol::serialize(std::ostream& sout) const
{
    // format version
    sout << "# Version" << std::endl;
    sout << 1 << std::endl;
    sout << "# Codec" << std::endl;
    sout << cdc << std::endl;
    sout << "# Security parameter eps_sec" << std::endl;
    sout << eps_sec << std::endl;
    sout << "# Correctness parameter eps_cor" << std::endl;
    sout << eps_cor << std::endl;
    sout << "# Alphabet size" << std::endl;
    sout << alphabet_size << std::endl;
    sout << "# Modem" << std::endl;
    sout << mdm << std::endl;
    sout << "# Mapper" << std::endl;
    sout << map << std::endl;

    return sout;
}

//! Deserialize protocol
std::istream&
dvqkd_protocol::serialize(std::istream& sin)
{
    assertalways(sin.good());

    // get format version
    int version;
    sin >> libbase::eatcomments >> version;

    // we have to serialise this as a codec object, then do a dynamic conversion
    std::shared_ptr<codec<libbase::vector>> _cdc;
    sin >> libbase::eatcomments >> _cdc >> libbase::verify;
    cdc = std::dynamic_pointer_cast<codec_coset<libbase::vector>>(_cdc);
    assert(cdc);

    // Verify casting was successful
    assert(this->cdc && "Loaded codec is not compatible with codec_coset!");

    // Created channel so it exists before seedfrom() is called.
    if (!this->demodulation_channel) {
        this->demodulation_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();
        this->demodulation_channel->set_parameter(0.0); // Default safe value
    }

    sin >> libbase::eatcomments >> eps_sec >> libbase::verify;
    sin >> libbase::eatcomments >> eps_cor >> libbase::verify;
    sin >> libbase::eatcomments >> alphabet_size >> libbase::verify;
    sin >> libbase::eatcomments >> mdm >> libbase::verify; 
    sin >> libbase::eatcomments >> map >> libbase::verify; 
    // check that all assumptions hold
    assertalways(cdc->num_inputs() == 2); // input has to be binary
    assertalways(cdc->num_outputs() == 2); // output has to be binary

    return sin;
}

const serializer dvqkd_protocol::shelper("qkd_protocol",
                                         "dvqkd_protocol",
                                         dvqkd_protocol::create);



} // namespace libcomm