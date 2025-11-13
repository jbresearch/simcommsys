/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_MODULE TestSyndromeAddition
#include <boost/test/included/unit_test.hpp>

#include "serializer_libcomm.h"
#include "codec/codec_coset.h"
#include "codec/ldpc.h"
#include "gf.h"
#include "matrix.h"
#include "channel/qsc.h"
#include "mapper/map_straight.h"
#include "modem.h"
#include "modem/direct_blockmodem.h"
#include "random.h"


/*!
 * \brief Boost unit tests to test out syndrome calculations and syndrome decoding. 
 * This will later be used in the BB84 protocol implementation. 
 */


// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#   undef DEBUG
#   define DEBUG 1
#endif

/**
 * @brief Helper function to print a GF_Q vector as ints
 */
template <class GFVec>
void print_gf_vector_as_ints(const GFVec& v)
{
    for (int i = 0; i < v.size(); ++i)
        std::cout << int(v(i)) << "\t";
    std::cout << std::endl;
}

/**
 * @brief Helper function to print a libbase::vector
 */
template<typename T>
void print_message(const std::string& title, const libbase::vector<T>& msg)
{
    std::cout << title << " (length " << msg.size() << "): [ ";
    // We assume the vector access is via operator() as shown
    // in your randperm.h/cpp files.
    for (int i = 0; i < msg.size(); ++i) {
        std::cout << msg(i) << " ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Helper function to create and configure an LPDC codec over GF2
 */
std::shared_ptr<libcomm::codec_coset<libbase::vector>> create_ldpc_codec_gf2() 
{
        
    std::stringstream cfg;
    cfg << R"SS(
# Version
5
# SPA type (trad|gdl)
gdl
# Number of iterations
50
# Clipping method
zero
# Value of almostzero
1e-100
# Reduce generator matrix to REF? (true|false)
1
# Length (n)
7
# Dimension (m)
4
# Max column weight
3
# Max row weight
4
# Non-zero values (ones|random|provided)
ones
# Column weight vector
7
3 3 3 1 1 1 1
# Row weight vector
4
3 3 4 3
# Non zero positions per col
3
1 3 4
3
1 2 3
3
2 3 4
1
1
1
2
1 
3
1
4
)SS";

    auto cdc = std::make_shared<libcomm::ldpc<libbase::gf2, double>>();
    cdc->serialize(cfg);
    return cdc;
}

/**
 * @brief Helper function to create and configure an LPDC codec over GF64
 */
std::shared_ptr<libcomm::codec_coset<libbase::vector>> create_ldpc_codec_gf64() 
{
    // Codec: ldpc<gf64,double>
    std::stringstream cfg;
    cfg << R"SS(
# Version
5
# SPA type (trad|gdl)
gdl
# Number of iterations
100
# Clipping method
zero
# Value of almostzero
1e-100
# Reduce generator matrix to REF? (true|false)
0
# Length (n)
16
# Dimension (m)
8
# Max column weight
2
# Max row weight
4
# Non-zero values (ones|random|provided)
random
# seed 
7
# Column weight vector
16
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# Row weight vector
8
4 4 4 4 4 4 4 4
# Non zero positions per col
2
4 6
2
3 7
2
5 8
2
1 2
2
2 7
2
3 6
2
1 3
2
4 7
2
2 5
2
1 8
2
4 5
2
6 8
2
7 8
2
3 5
2
2 6
2
1 4
)SS";

    auto cdc = std::make_shared<libcomm::ldpc<libbase::gf64, double>>();
    cdc->serialize(cfg);
    return cdc;
}

/**
 * @brief Helper function to create and configure a direct block modem over GF2
 */
std::shared_ptr<libcomm::blockmodem<libbase::gf2>> create_modem_gf2() 
{
        
 std::stringstream cfg;
cfg << R"SS(
direct_blockmodem<gf2,vector,double>
)SS";

    auto mdm = std::make_shared<libcomm::direct_blockmodem<libbase::gf2, libbase::vector, double>>();
    mdm->serialize(cfg);
    return mdm;
}


/**
 * @brief Helper function to create and configure a direct block modem over GF64
 */
std::shared_ptr<libcomm::blockmodem<libbase::gf64>> create_modem_gf64() 
{
        
 std::stringstream cfg;
cfg << R"SS(
direct_blockmodem<gf64,vector,double>
)SS";

    auto mdm = std::make_shared<libcomm::direct_blockmodem<libbase::gf64, libbase::vector, double>>();
    mdm->serialize(cfg);
    return mdm;
}

/**
 * @brief Generates a random message of a given length and GF size.
 *
 * @tparam T The element type of the vector (e.g., int, libbase::gf2, libbase::gf16)
 * @param k The desired message length.
 * @param m The size of the Galois Field (e.g., 2 for GF(2), 16 for GF(16)).
 * @param rng A reference to an initialized random generator.
 * @return A libbase::vector<T> containing the random message.
 */
template<typename T>
libbase::vector<T> generate_random_message(int k, uint32_t m, libbase::randgen& rng)
{
    libbase::vector<T> message;
    message.init(k); // Initialize vector to size k
    
    for (int i = 0; i < k; ++i) {
        // ival(m) generates a random int in [0, m-1]
        // The constructor for T (e.g., gf16(int)) should handle the conversion.
        message(i) = T(rng.ival(m));
    }
    return message;
}

/**
 * @brief Generates a random codeword of a given length and GF size.
 *
 * @tparam T The element type of the vector (e.g., int, libbase::gf2, libbase::gf16)
 * @param k The desired codeword length.
 * @param m The size of the Galois Field (e.g., 2 for GF(2), 16 for GF(16)).
 * @param rng A reference to an initialized random generator.
 * @return A libbase::vector<T> containing the random codeword.
 */
template<typename T>
libbase::vector<T> generate_codeword(int n, uint32_t m, libbase::randgen& rng)
{
    libbase::vector<T> codeword;
    codeword.init(n); // Initialize codeword to size k
    
    for (int i = 0; i < n; ++i) {
        // ival(m) generates a random int in [0, m-1]
        // The constructor for T (e.g., gf16(int)) should handle the conversion.
        codeword(i) = T(rng.ival(m));
    }
    return codeword;
}

BOOST_AUTO_TEST_CASE(gf2_victor_example_no_errors)
{
    /* Test 1a - Generates a codeword by encoding a message of size k.
    Calculate syndrome which will still be all zero. 
    Modulate and Demodulate using direct block modem. 
    Pass the codeword over a qsc channel with no noise (ps = 0). 
    Decode codeword using original all zero syndrome. 
    */

    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << std::endl << "******* Boost Test 1a *******" << std::endl; 

    // Create LDPC codec over GF2 
    auto cdc = create_ldpc_codec_gf2();
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
#endif

    // Create and seed the random generator
    libbase::randgen rng;
    const int seed_number = 12345;
    rng.seed(seed_number);
   

    const int k = cdc->input_block_size();

    /* Known test from Victor's notes, u = [101], v = [1, 0, 1, 1, 1, 0, 0]
    without errors */
    libbase::vector<int> original_message;
    original_message.init(k); 
    original_message(0) = 1; 
    original_message(1) = 0;
    original_message(2) = 1; 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Original Message: " << original_message << std::endl;
#endif

    // Encode message to get codeword
    const int n = cdc->output_block_size();
    libbase::vector<int> generated_codeword(n);
    cdc->encode(original_message, generated_codeword);
    print_message("Generated codeword = ", generated_codeword);

    // Calculate syndrome of generated codeword which will be sent over c. channel
    libbase::vector<int> calculated_syndrome;
    cdc->calculate_syndrome(generated_codeword, calculated_syndrome);
    print_message("Generated syndrome = ", calculated_syndrome);

    /* Modulate codeword */
    auto mdm = create_modem_gf2();

#if DEBUG >= 1
std::cout << "TESTSYNDROMEDECODING: Modem Details: " << mdm->description() << std::endl;
#endif

    mdm->set_blocksize(libbase::size_type<libbase::vector>(n));

    const int M = mdm->num_symbols(); 
    std::cout << "Size M = " << M << std::endl;

    libbase::vector<libbase::gf2> modulated_codeword(7);

    // Call modulate with the 3 required arguments:
    //  (int symbol_count, vector<int>& input, vector<gf2>& output)
    mdm->modulate(M, generated_codeword, modulated_codeword); 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Modulated codeword: " << std::endl;
    // Assuming you have a print_message for gf2 or similar
    print_gf_vector_as_ints(modulated_codeword); 
#endif

    /* Transmit codeword through a QSC channel */
    // Initialise channel 
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

    // Probability of Substitution, Ps
    double Ps = 0.0; 
    qsc_channel->set_parameter(Ps);

    // Seed the channel
    qsc_channel->seedfrom(rng);

#if DEBUG >= 1
    std::cerr << "TESTSYNDROMEDECODING: Details of QSC Channel: "
                  << qsc_channel ->description() << std::endl;
    std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps << std::endl; // To do: ideally you get the channel parameter directly from the channel itself
#endif

    libbase::vector<libbase::gf2> received_codeword(7);
    qsc_channel->transmit(modulated_codeword, received_codeword);

    std::cout << "Received GF2 codeword:  " << std::endl; 
    print_gf_vector_as_ints(received_codeword);

    // Initialise probability table.
    auto prob_table = libbase::vector<libbase::vector<double>>(n);

    /* Demodulate the received codeword */
    mdm->demodulate(*qsc_channel, received_codeword, prob_table);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Probability Table: "
                  << prob_table << std::endl;
#endif

    // Seed the codec
    cdc->seedfrom(rng);

    /* Decode the demodulated codeword to get the message using the original syndrome*/
    cdc->init_decoder(prob_table, calculated_syndrome);

    auto decoded_message_u = libbase::vector<int>(cdc->input_block_size());
    cdc->decode(decoded_message_u);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u: "
                  << decoded_message_u << std::endl;
#endif

    /* Validation of result */
    // Compute number of errors using hamming distance. 
    int num_errors =  libbase::hamming(original_message, decoded_message_u);
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Number of errors between original and decoded messages: " << num_errors << std::endl;
#endif

    // Check that the decoded and original messages are the same 
    BOOST_CHECK_EQUAL(num_errors, 0); 
}

BOOST_AUTO_TEST_CASE(gf2_victor_with_errors)
{
    /* Test 1b - Generates a codeword by encoding a message of size k 
    Calculate syndrome which will still be all zero. 
    Modulate and Demodulate using direct block modem. 
    Pass the codeword over a qsc channel
    with ps = 0.1. Decode codeword using original all zero syndrome. 
    */

    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << std::endl << std::endl << "******* Boost Test 1b *******" << std::endl; 

    // Create codec
    auto cdc = create_ldpc_codec_gf2();
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
#endif

    // Create and seed the random generator
    libbase::randgen rng;
    const int seed_number = 12345;
    rng.seed(seed_number);

    const int k = cdc->input_block_size();

    // Known test from Victor's notes, u = [101], v = [1, 0, 1, 1, 1, 0, 0]
    libbase::vector<int> original_message;
    original_message.init(k); 
    original_message(0) = 1; 
    original_message(1) = 0;
    original_message(2) = 1; 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Original message: " << original_message << std::endl;
#endif

    // Encode message to get codeword
    const int n = cdc->output_block_size();
    libbase::vector<int> generated_codeword(n);
    cdc->encode(original_message, generated_codeword);

#if DEBUG >= 1
std::cout << "TESTSYNDROMEDECODING: Generate codeword: " << generated_codeword << std::endl;
#endif

    // Calculate syndrome of generated codeword which will be sent over CC
    libbase::vector<int> calculated_syndrome;
    cdc->calculate_syndrome(generated_codeword, calculated_syndrome);

#if DEBUG >= 1
std::cout << "TESTSYNDROMEDECODING: Generated syndrome: " << calculated_syndrome << std::endl;
#endif

    /* Modulate codeword */

    // Create modem 
    auto mdm = create_modem_gf2();

#if DEBUG >= 1
std::cout << "TESTSYNDROMEDECODING: Modem Details: " << mdm->description() << std::endl;
#endif

    mdm->set_blocksize(libbase::size_type<libbase::vector>(n));

    const int M = mdm->num_symbols(); 
    std::cout << "Size M = " << M << std::endl;

    libbase::vector<libbase::gf2> modulated_codeword(n);

    // Call modulate with the 3 required arguments:
    //    (int symbol_count, vector<int>& input, vector<gf2>& output)
    mdm->modulate(M, generated_codeword, modulated_codeword); 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Modulated codeword: " << std::endl;
    print_gf_vector_as_ints(modulated_codeword); 
#endif

    /* Transmit codeword through a QSC channel */
    // Create channel 
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

    // Probability of Substitution, Ps
    double Ps = 0.1; 
    // Set Ps
    qsc_channel->set_parameter(Ps);

    // Seed the channel
    qsc_channel->seedfrom(rng);

#if DEBUG >= 1
    std::cerr << "TESTSYNDROMEDECODING: Details of QSC Channel: "
                  << qsc_channel ->description() << std::endl;
    std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps << std::endl; 
#endif

    libbase::vector<libbase::gf2> received_codeword(7);
    qsc_channel->transmit(modulated_codeword, received_codeword);
    std::cout << "Received Corrupted GF2 codeword:  " << std::endl; 
    print_gf_vector_as_ints(received_codeword);
  
    // Initialise probability table.
    auto prob_table = libbase::vector<libbase::vector<double>>(n);

    /* Demodulate the received codeword */
    mdm->demodulate(*qsc_channel, received_codeword, prob_table);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Probability Table: "
                  << prob_table << std::endl;
#endif

    // Seed the codec
    cdc->seedfrom(rng);

    /* Decode the demodulated codeword to get the message using the original syndrome*/
    cdc->init_decoder(prob_table, calculated_syndrome);

    auto decoded_message_u = libbase::vector<int>(cdc->input_block_size());
    cdc->decode(decoded_message_u);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u: "
                  << decoded_message_u << std::endl;
#endif

    /* Validation of result */
    // Compute number of errors using hamming distance
    int num_errors =  libbase::hamming(original_message, decoded_message_u);
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Number of errors between original and decoded messages: " << num_errors <<  std::endl;
#endif
    // Check that the decoded and original messages are the same 
    BOOST_CHECK_EQUAL(num_errors, 0); 
}
    
BOOST_AUTO_TEST_CASE(gf2_random_codeword_comparison)
{
/* * This single test case combines the logic from the two 'gf2_random_codeword' tests.
 *
 * Part 1 (Test 2a): Generates a random codeword, calculates its syndrome,
 * and decodes it with NO noise.
 *
 * Part 2 (Test 2b): Reuses the *same* codeword and syndrome from Part 1,
 * passes the codeword through a noisy channel, and decodes
 * it using the original syndrome.
 *
 * Finally, it compares the two decoded messages.
 */

    const libcomm::serializer_libcomm my_serializer_libcomm;
    std::cout << std::endl << "******* Boost Test 2a/2b - Generate random GF2 codeword and syndrome decode (no-noise vs. noise) *******" << std::endl; 

    auto cdc = create_ldpc_codec_gf2();
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
#endif

    // Create and seed the random generator
    libbase::randgen rng;
    const int seed_number = 12345;
    rng.seed(seed_number);

    // Alphabet size 
    const int q = 2;
    const int n = cdc->output_block_size();
    
    // These are the variables that will be shared
    libbase::vector<int> generated_codeword(n);
    libbase::vector<int> calculated_syndrome; 
    // Note: Syndrome size is typically m = n-k, but codec_coset seems to use n.
    auto decoded_message_u_no_error = libbase::vector<int>(cdc->input_block_size());

    // Create modem and channel
    auto mdm = create_modem_gf2();
    mdm->set_blocksize(libbase::size_type<libbase::vector>(n));
    const int M = mdm->num_symbols();
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();


    // ####################################################################
    // ## PART 1: No Errors (gf2_random_codeword_without_errors)
    // ####################################################################
    std::cout << "\n--- PART 1: No Noise Test ---" << std::endl;

    // Randomly generate a codeword of size n
    generated_codeword = generate_codeword<int>(n, q, rng);
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Generated codeword: " << generated_codeword << std::endl;
#endif

    // Calculate syndrome of generated codeword
    cdc->calculate_syndrome(generated_codeword, calculated_syndrome);
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Generated syndrome: " << calculated_syndrome << std::endl;
#endif

    // Modulate codeword
    libbase::vector<libbase::gf2> modulated_codeword_p1(n);
    mdm->modulate(M, generated_codeword, modulated_codeword_p1); 

    // Transmit codeword through a QSC channel with Ps = 0.0
    double Ps_no_error = 0.0; 
    qsc_channel->set_parameter(Ps_no_error);
    qsc_channel->seedfrom(rng); 

    libbase::vector<libbase::gf2> received_codeword_p1(n);
    qsc_channel->transmit(modulated_codeword_p1, received_codeword_p1);
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Received GF2 codeword (P_s = 0.0): " << std::endl;
    print_gf_vector_as_ints(received_codeword_p1);
#endif

    // Demodulate
    auto prob_table_p1 = libbase::vector<libbase::vector<double>>(n);
    mdm->demodulate(*qsc_channel, received_codeword_p1, prob_table_p1);

    // Seed and decode
    cdc->seedfrom(rng); // Re-seed for deterministic test
    cdc->init_decoder(prob_table_p1, calculated_syndrome);
    cdc->decode(decoded_message_u_no_error);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u (no error): "
              << decoded_message_u_no_error << std::endl;
#endif

    // ####################################################################
    // ## PART 2: With Errors (gf2_random_codeword_with_errors)
    // ####################################################################
    std::cout << "\n--- PART 2: With Noise Test ---" << std::endl;

    // REUSE 'generated_codeword' and 'calculated_syndrome' from Part 1.
    // Modulate codeword (same as before)
    libbase::vector<libbase::gf2> modulated_codeword_p2(n);
    mdm->modulate(M, generated_codeword, modulated_codeword_p2); 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Modulated codeword (for noisy channel): " << std::endl;
    print_gf_vector_as_ints(modulated_codeword_p2); 
#endif

    // Transmit codeword through a QSC channel with Ps = 0.1
    double Ps_with_error = 0.1; 
    qsc_channel->set_parameter(Ps_with_error);
    qsc_channel->seedfrom(rng); 

#if DEBUG >= 1
    std::cerr << "TESTSYNDROMEDECODING: Details of QSC Channel: "
              << qsc_channel ->description() << std::endl;
    std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps_with_error << std::endl;
#endif

    libbase::vector<libbase::gf2> received_codeword_p2(n);
    qsc_channel->transmit(modulated_codeword_p2, received_codeword_p2);

    std::cout << "Received GF2 codeword (P_s=0.1): "; 
    print_gf_vector_as_ints(received_codeword_p2);
 
    // Demodulate
    auto prob_table_p2 = libbase::vector<libbase::vector<double>>(n);
    mdm->demodulate(*qsc_channel, received_codeword_p2, prob_table_p2);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Probability Table (noisy): "
              << prob_table_p2 << std::endl;
#endif

    // Seed and decode
    cdc->seedfrom(rng); // Re-seed for deterministic test
    
    // Key Step: Use the prob_table from the noisy channel, but the
    // syndrome calculated from the *original* noiseless codeword.
    cdc->init_decoder(prob_table_p2, calculated_syndrome);

    auto decoded_message_u_with_error = libbase::vector<int>(cdc->input_block_size());
    cdc->decode(decoded_message_u_with_error);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u (with error): "
              << decoded_message_u_with_error << std::endl;
#endif

    // ####################################################################
    // ## PART 3: Comparison
    // ####################################################################
    std::cout << "\n--- PART 3: Comparison ---" << std::endl;
    
    // Compute number of errors between the two decoded messages
    int num_errors = libbase::hamming(decoded_message_u_no_error, decoded_message_u_with_error);
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Validating Result " << std::endl;
    std::cout << "TESTSYNDROMEDECODING: Decoded message (no error): " << decoded_message_u_no_error << std::endl;
    std::cout << "TESTSYNDROMEDECODING: Decoded message (with error): " << decoded_message_u_with_error << std::endl;
    std::cout << "TESTSYNDROMEDECODING: Number of errors between decoded messages: " << num_errors << std::endl;
#endif 

    // Assert that the syndrome decoding successfully corrected the channel errors.
    BOOST_CHECK_EQUAL(num_errors, 0); 
}

/* * This test case runs the entire simulation in a loop 100 times, 
 * using a different seed for each run.
 *
 * It counts the number of times the decoder SUCCEEDS (correctly
 * decodes the message despite noise) and the number of times it
 * FAILS (the noise causes a decoding error).
 */
BOOST_AUTO_TEST_CASE(gf2_random_codeword_loop)
{
    const libcomm::serializer_libcomm my_serializer_libcomm;
    std::cout << std::endl << "******* Boost Test 3 - Looping syndrome decode n times over GF2 *******" << std::endl; 

    auto cdc = create_ldpc_codec_gf2();
    auto mdm = create_modem_gf2();
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

    // Create and seed the random generator
    libbase::randgen rng;
    // Starting seed
    const int base_seed_number = 12345; 

    // Alphabet size 
    const int q = 2;
    const int n = cdc->output_block_size();
    
    mdm->set_blocksize(libbase::size_type<libbase::vector>(n));
    const int M = mdm->num_symbols();

    int num_successes = 0;
    int num_failures = 0;
    const int TOTAL_RUNS = 20;

    for (int i = 0; i < TOTAL_RUNS; ++i) 
    {
        // --- Each loop gets a new, unique seed ---
        const int current_seed = base_seed_number + i;
        rng.seed(current_seed);
        
        std::cout << "\n--- RUN " << i << " (Seed: " << current_seed << ") ---" << std::endl;

        // --- These are the variables for this loop iteration ---
        libbase::vector<int> generated_codeword(n);
        libbase::vector<int> calculated_syndrome;
        auto decoded_message_u_no_error = libbase::vector<int>(cdc->input_block_size());
        auto decoded_message_u_with_error = libbase::vector<int>(cdc->input_block_size());

        // ####################################################################
        // ## PART 1: No Errors (Ground Truth)
        // ####################################################################
        
        generated_codeword = generate_codeword<int>(n, q, rng);
        cdc->calculate_syndrome(generated_codeword, calculated_syndrome);

        // Modulate -> Channel (Ps=0) -> Demodulate
        libbase::vector<libbase::gf2> modulated_codeword_p1(n);
        mdm->modulate(M, generated_codeword, modulated_codeword_p1); 
        
        qsc_channel->set_parameter(0.0);
        qsc_channel->seedfrom(rng);
        
        libbase::vector<libbase::gf2> received_codeword_p1(n);
        qsc_channel->transmit(modulated_codeword_p1, received_codeword_p1);
    
        auto prob_table_p1 = libbase::vector<libbase::vector<double>>(n);
        mdm->demodulate(*qsc_channel, received_codeword_p1, prob_table_p1);

        // Decode
        cdc->seedfrom(rng);
        cdc->init_decoder(prob_table_p1, calculated_syndrome);
        cdc->decode(decoded_message_u_no_error);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u (no error): "
              << decoded_message_u_no_error << std::endl;
#endif
    
        // ####################################################################
        // ## PART 2: With Errors (Added Noise)
        // ####################################################################

        // Modulate -> Channel (Ps=0.1) -> Demodulate
        libbase::vector<libbase::gf2> modulated_codeword_p2(n);
        mdm->modulate(M, generated_codeword, modulated_codeword_p2); 

        double Ps_with_error = 0.1; // 10% errors
        // double Ps_with_error = 0.05; // 5% noise 
        // double Ps_with_error = 0.02; // 2% noise 
        // double Ps_with_error = 0.01; // 1% noise 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Ps = "
              << Ps_with_error << std::endl;
#endif

        qsc_channel->set_parameter(Ps_with_error);
        qsc_channel->seedfrom(rng);

        // corrupted codeword 
        libbase::vector<libbase::gf2> received_codeword_p2(n); 
        qsc_channel->transmit(modulated_codeword_p2, received_codeword_p2);

        auto prob_table_p2 = libbase::vector<libbase::vector<double>>(n);
        mdm->demodulate(*qsc_channel, received_codeword_p2, prob_table_p2);
    
        // Decode (using original syndrome)
        cdc->seedfrom(rng);
        cdc->init_decoder(prob_table_p2, calculated_syndrome);
        cdc->decode(decoded_message_u_with_error);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message (with error): "
              << decoded_message_u_with_error << std::endl;
#endif

        // ####################################################################
        // ## PART 3: Comparison
        // ####################################################################
        int num_errors = libbase::hamming(decoded_message_u_no_error, decoded_message_u_with_error);
        
        if (num_errors == 0) {
            std::cout << ">>> RESULT: SUCCESS (Messages match)" << std::endl;
            num_successes++;
        } else {
            std::cout << ">>> RESULT: FAILURE (Errors: " << num_errors << ")" << std::endl;
            num_failures++;
        }
    }

    // ####################################################################
    // ## FINAL RESULTS
    // ####################################################################
    std::cout << std::endl << std::endl << "--- FINAL SIMULATION RESULTS ---" << std::endl;
    std::cout << "Total Runs:  " << TOTAL_RUNS << std::endl;
    std::cout << "Successes:   " << num_successes << std::endl;
    std::cout << "Failures:    " << num_failures << std::endl;
    
    BOOST_CHECK_EQUAL(num_successes + num_failures, TOTAL_RUNS);
    BOOST_CHECK(num_successes > 0); // Check that the decoder *can* succeed
    BOOST_CHECK(num_failures > 0);  // Check that the channel *does* cause errors
}

BOOST_AUTO_TEST_CASE(gf64_encoded_codeword_comparison)
{
    const libcomm::serializer_libcomm my_serializer_libcomm;
    std::cout << std::endl << "******* Boost Test 4 - Generate GF64 codeword from random message and syndrome decode (with noise) *******" << std::endl; 

    auto cdc = create_ldpc_codec_gf64();
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
#endif

    // Create and seed the random generator
    libbase::randgen rng;
    const int seed_number = 12345;
    rng.seed(seed_number);

    // Alphabet size 
    const int q = 64;
    
    // Size of original message/information bits
    const int k = cdc->input_block_size();
    // Size of encode message/codeword
    const int n = cdc->output_block_size();

    // Initialise with size k 
    libbase::vector<int> original_message(k);
    // Randomly generate message u with size k
    original_message = generate_random_message<int>(k, q, rng);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: original message u: "
                  << original_message << std::endl;
#endif

    // Encode u to get codeword v of size n
    libbase::vector<int> generated_codeword(n);
    cdc->encode(original_message, generated_codeword);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Generated codeword: " << generated_codeword << std::endl;
#endif

    // Calculate syndrome of generated codeword which will be sent over CC
    libbase::vector<int> calculated_syndrome; // Size of n-k 
    cdc->calculate_syndrome(generated_codeword, calculated_syndrome);
    print_message("Generated syndrome = ", calculated_syndrome);

    /* Modulate codeword */
    auto mdm = create_modem_gf64();

#if DEBUG >= 1
std::cout << "TESTSYNDROMEDECODING: Modem Details: " << mdm->description() << std::endl;
#endif

    mdm->set_blocksize(libbase::size_type<libbase::vector>(n));

    const int M = mdm->num_symbols(); 
    std::cout << "Size M = " << M << std::endl;

    libbase::vector<libbase::gf64> modulated_codeword(n);

    // Call modulate with the 3 required arguments:
    //    (int symbol_count, vector<int>& input, vector<gf2>& output)
    mdm->modulate(M, generated_codeword, modulated_codeword); 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Modulated codeword: " << std::endl;
    print_gf_vector_as_ints(modulated_codeword); 
#endif

    /* Transmit codeword through a QSC channel */
    // Initialise channel 
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf64>>();

    // Probability of Substitution, Ps
    double Ps = 0.1; 
    // Set Ps 
    qsc_channel->set_parameter(Ps);

    // Seed the channel
    qsc_channel->seedfrom(rng);

#if DEBUG >= 1
    std::cerr << "TESTSYNDROMEDECODING: Details of QSC Channel: "
                  << qsc_channel ->description() << std::endl;
    std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps << std::endl; 
#endif

    libbase::vector<libbase::gf64> received_codeword(n);
    qsc_channel->transmit(modulated_codeword, received_codeword);

#if DEBUG >= 1
    std::cout << "Received Corrupted GF64 codeword:  " << std::endl; 
    print_gf_vector_as_ints(received_codeword);
#endif

    // Initialise probability table
    auto prob_table = libbase::vector<libbase::vector<double>>(n);

    /* Demodulate the received codeword */
    mdm->demodulate(*qsc_channel, received_codeword, prob_table);

// #if DEBUG >= 1
//     std::cout << "TESTSYNDROMEDECODING: Probability Table: "
//                   << prob_table << std::endl;
// #endif

    // Seed the codec
    cdc->seedfrom(rng);

    /* Decode the demodulated codeword to get the message using the original syndrome*/
    cdc->init_decoder(prob_table, calculated_syndrome);

    auto decoded_message_u = libbase::vector<int>(cdc->input_block_size());
    cdc->decode(decoded_message_u);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u: "
                  << decoded_message_u << std::endl;
#endif

    // Compute number of errors using hamming distance
    int num_errors =  libbase::hamming(original_message, decoded_message_u);
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Number of errors between original and decoded messages: " << num_errors <<  std::endl;
#endif
    // Check that the decoded message and the original message are the same.
    BOOST_CHECK_EQUAL(num_errors, 0); 
}

BOOST_AUTO_TEST_CASE(gf64_random_codeword_loop)
{

/* * This test case runs the entire simulation over GF64 in a loop 100 times, 
 * using a different seed for each run.
 *
 * It counts the number of times the decoder SUCCEEDS (correctly
 * decodes the message despite noise) and the number of times it
 * FAILS (the noise causes a decoding error).
 */

    const libcomm::serializer_libcomm my_serializer_libcomm;
    std::cout << std::endl << "******* Boost Test 5 - Looping syndrome decode n times over GF64 *******" << std::endl; 

    auto cdc = create_ldpc_codec_gf64();
    auto mdm = create_modem_gf64();
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf64>>();

    // Create and seed the random generator
    libbase::randgen rng;
    // Starting seed
    const int base_seed_number = 12345; 

    // Alphabet size 
    const int q = 64;
    const int n = cdc->output_block_size();
    // const int k = cdc->input_block_size();
    
    mdm->set_blocksize(libbase::size_type<libbase::vector>(n));
    const int M = mdm->num_symbols();

    int num_successes = 0;
    int num_failures = 0;
    const int TOTAL_RUNS = 20;

    for (int i = 0; i < TOTAL_RUNS; ++i) 
    {
        // --- Each loop gets a new, unique seed ---
        const int current_seed = base_seed_number + i;
        rng.seed(current_seed);
        
        std::cout << "\n--- RUN " << i << " (Seed: " << current_seed << ") ---" << std::endl;

        // --- These are the variables for this loop iteration ---
        libbase::vector<int> generated_codeword(n);
        libbase::vector<int> calculated_syndrome; // Size of n-k
        auto decoded_message_u_no_error = libbase::vector<int>(cdc->input_block_size());
        auto decoded_message_u_with_error = libbase::vector<int>(cdc->input_block_size());

        // ####################################################################
        // ## PART 1: No Errors (Ground Truth)
        // ####################################################################
        
        generated_codeword = generate_codeword<int>(n, q, rng);
        cdc->calculate_syndrome(generated_codeword, calculated_syndrome);

        // Modulate -> Channel (Ps=0) -> Demodulate
        libbase::vector<libbase::gf64> modulated_codeword_p1(n);
        mdm->modulate(M, generated_codeword, modulated_codeword_p1); 
        
        qsc_channel->set_parameter(0.0);
        qsc_channel->seedfrom(rng);
        
        libbase::vector<libbase::gf64> received_codeword_p1(n);
        qsc_channel->transmit(modulated_codeword_p1, received_codeword_p1);
    
        auto prob_table_p1 = libbase::vector<libbase::vector<double>>(n);
        mdm->demodulate(*qsc_channel, received_codeword_p1, prob_table_p1);

        // Decode
        cdc->seedfrom(rng);
        cdc->init_decoder(prob_table_p1, calculated_syndrome);
        cdc->decode(decoded_message_u_no_error);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message u (no error): "
              << decoded_message_u_no_error << std::endl;
#endif
    
        // ####################################################################
        // ## PART 2: With Errors (Added Noise)
        // ####################################################################

        // Modulate -> Channel (Ps=0.1) -> Demodulate
        libbase::vector<libbase::gf64> modulated_codeword_p2(n);
        mdm->modulate(M, generated_codeword, modulated_codeword_p2); 

        // double Ps_with_error = 0.1; // 10% noise 
        double Ps_with_error = 0.05; // 5% noise 
        // double Ps_with_error = 0.02; // 2% noise 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Ps = "
              << Ps_with_error << std::endl;
#endif

        qsc_channel->set_parameter(Ps_with_error);
        qsc_channel->seedfrom(rng);

        // corrupted codeword 
        libbase::vector<libbase::gf64> received_codeword_p2(n); 
        qsc_channel->transmit(modulated_codeword_p2, received_codeword_p2);

        auto prob_table_p2 = libbase::vector<libbase::vector<double>>(n);
        mdm->demodulate(*qsc_channel, received_codeword_p2, prob_table_p2);
    
        // Decode (using original syndrome)
        cdc->seedfrom(rng);
        cdc->init_decoder(prob_table_p2, calculated_syndrome);
        cdc->decode(decoded_message_u_with_error);

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Decoded message (with error): "
              << decoded_message_u_with_error << std::endl;
#endif

        // ####################################################################
        // ## PART 3: Comparison
        // ####################################################################
        int num_errors = libbase::hamming(decoded_message_u_no_error, decoded_message_u_with_error);
        
        if (num_errors == 0) {
            std::cout << ">>> RESULT: SUCCESS (Messages match)" << std::endl;
            num_successes++;
        } else {
            std::cout << ">>> RESULT: FAILURE (Errors: " << num_errors << ")" << std::endl;
            num_failures++;
        }
    }

    // ####################################################################
    // ## FINAL RESULTS
    // ####################################################################
    std::cout << std::endl << std::endl << "--- FINAL SIMULATION RESULTS ---" << std::endl;
    std::cout << "Total Runs:  " << TOTAL_RUNS << std::endl;
    std::cout << "Successes:   " << num_successes << std::endl;
    std::cout << "Failures:    " << num_failures << std::endl;
    
    BOOST_CHECK_EQUAL(num_successes + num_failures, TOTAL_RUNS);
    BOOST_CHECK(num_successes > 0); // Check that the decoder *can* succeed
    BOOST_CHECK(num_failures > 0);  // Check that the channel *does* cause errors
}