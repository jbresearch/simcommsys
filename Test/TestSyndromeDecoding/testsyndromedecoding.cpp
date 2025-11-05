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

template <class GFVec>
void
print_gf_vector_as_ints(const GFVec& v)
{
    for (int i = 0; i < v.size(); ++i)
        std::cout << int(v(i));
    std::cout << "\n";
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

// Helper function to create and configure the codec
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
7
# Max column weight
3
# Max row weight
3
# Non-zero values (ones|random|provided)
ones
# Column weight vector
7
3 3 3 3 3 3 3
# Row weight vector
7
3 3 3 3 3 3 3
# Non zero positions per col
3
1 5 7
3
1 2 6
3
2 3 7
3
1 3 4
3
2 4 5
3
3 5 6
3
4 6 7
)SS";

    auto cdc = std::make_shared<libcomm::ldpc<libbase::gf2, double>>();
    cdc->serialize(cfg);
    return cdc;
}

// Helper function to create and configure the codec
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
    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << "Boost Test 1a" << std::endl; 

    auto cdc = create_ldpc_codec_gf2();
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
#endif

    // Create and seed the random generator
    libbase::randgen rng;
    const int seed_number = 12345;
    rng.seed(seed_number);


    /* Test 1 - Generates a codeword by encoding a message of size k.
    Calculate syndrome which will still be all zero. 
    Modulate and Demodulate using direct block modem. 
    Pass the codeword over a qsc channel with no noise (ps = 0). 
    Decode codeword using original all zero syndrome. 
    */

    const int k = cdc->input_block_size();

    /* Known test from Victor's notes, u = [101], v = [1, 0, 1, 1, 1, 0, 0]
    without errors */
    libbase::vector<int> original_message;
    original_message.init(k); 
    original_message(0) = 1; 
    original_message(1) = 0;
    original_message(2) = 1; 

    // Encode message to get codeword
    const int n = cdc->output_block_size();
    libbase::vector<int> generated_codeword(n);
    cdc->encode(original_message, generated_codeword);
    print_message("Generated codeword = ", generated_codeword);

    // Calculate syndrome of generated codeword which will be sent over c. channel
    libbase::vector<int> calculated_syndrome(n);
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
    libbase::vector<int> validation_codeword(n);
    cdc->encode(decoded_message_u, validation_codeword);
    print_message("Validation codeword = ", validation_codeword);

    // Compute number of errors using hamming distance
    int num_errors =  libbase::hamming(original_message, decoded_message_u);
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Validating Result " << std::endl;
    std::cout << "TESTSYNDROMEDECODING: Encode decoded message: " << validation_codeword <<  std::endl;
    std::cout << "TESTSYNDROMEDECODING: Number of errors between original and decoded messages: " << num_errors << std::endl;
#endif
}

BOOST_AUTO_TEST_CASE(gf2_victor_with_errors)
{
    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << "Boost Test 1b" << std::endl; 

    auto cdc = create_ldpc_codec_gf2();
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
#endif

    // Create and seed the random generator
    libbase::randgen rng;
    const int seed_number = 12345;
    rng.seed(seed_number);

    /* Test 2 - Generates a codeword by encoding a message of size k 
    Calculate syndrome which will still be all zero. 
    Modulate and Demodulate using direct block modem. Pass the codeword over a qsc channel
    with no noise (ps = 0). Pass the codeword over a qsc channel
    with ps = 0.1. Decode codeword using original all zero syndrome. 
    */

    const int k = cdc->input_block_size();

    // Known test from Victor's notes, u = [101], v = [1, 0, 1, 1, 1, 0, 0]
    libbase::vector<int> original_message;
    original_message.init(k); 
    original_message(0) = 1; 
    original_message(1) = 0;
    original_message(2) = 1; 

    // Encode message to get codeword
    const int n = cdc->output_block_size();
    libbase::vector<int> generated_codeword(n);
    cdc->encode(original_message, generated_codeword);
    print_message("Generated codeword = ", generated_codeword);

    // Calculate syndrome of generated codeword which will be sent over CC
    libbase::vector<int> calculated_syndrome(n);
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
    //    (int symbol_count, vector<int>& input, vector<gf2>& output)
    mdm->modulate(M, generated_codeword, modulated_codeword); 

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Modulated codeword: " << std::endl;
    // Assuming you have a print_message for gf2 or similar
    print_gf_vector_as_ints(modulated_codeword); 
#endif

    /* Transmit codeword through a QSC channel */
    // Define channel parameters
    auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

    // Probability of Substitution, Ps
    double Ps = 0.1; 
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
    libbase::vector<int> validation_codeword(n);
    cdc->encode(decoded_message_u, validation_codeword);
    print_message("Validation codeword = ", validation_codeword);

    // Compute number of errors using hamming distance
    int num_errors =  libbase::hamming(original_message, decoded_message_u);
    
#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Validating Result " << std::endl;
    std::cout << "TESTSYNDROMEDECODING: Encode decoded message: " << validation_codeword << std::endl;
    std::cout << "TESTSYNDROMEDECODING: Number of errors between original and decoded messages: " << num_errors <<  std::endl;
#endif
}

/* Next test to add:
Test 2a: 
- Randomly generate a codeword (not the message vector).
- Calculate the syndrome based on this codeword without noise.
- Decode codeword without any noise to get the decoded message.

Test 2b: 
- Pass random codeword through a qsc channel. 
- Modulate 
- Demodulate codeword
- Decoding codeword using original calculated syndrome.
- Compare decoded message to the decoded message of test 3a. 

Test 3: Repeat test 3 for GFQ e.g. GF16 
*/

// BOOST_AUTO_TEST_CASE(gf2_victor_with_errors)
// {
//     const libcomm::serializer_libcomm my_serializer_libcomm;
//     // std::cout << "Randomly generate a codeword and calculate syndrome over GF2" << std::endl; 

//     auto cdc = create_ldpc_codec_gf2();
    
// #if DEBUG >= 1
//     std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description() << std::endl;
// #endif

//     // Create and seed the random generator
//     libbase::randgen rng;
//     const int seed_number = 12345;
//     rng.seed(seed_number);

//     // Alphabet size 
//     // const int q = 2;  

//     /* Test 1 - Generates a codeword by encoding a message of size k, Syndrome will be all zeros. 
//     // Define your desired message length    */
//     const int k = cdc->input_block_size();

//     // libbase::vector<int> original_message = generate_random_message<int>(k, q, rng);
//     // print_message("Random message = ", original_message);

//     // Known test from Victor's notes, u = [101], v = [1, 0, 1, 1, 1, 0, 0] - Test 2
//     libbase::vector<int> original_message;
//     original_message.init(k); 
//     original_message(0) = 1; 
//     original_message(1) = 0;
//     original_message(2) = 1; 

//     // Encode message to get codeword
//     const int n = cdc->output_block_size();
//     libbase::vector<int> generated_codeword(n);
//     cdc->encode(original_message, generated_codeword);
//     print_message("Generated codeword = ", generated_codeword);

//     // // Randomly generate a codeword of size n - Test 3 
//     // const int n = cdc->output_block_size();
//     // libbase::vector<int> generated_codeword(n);
//     // generated_codeword = generate_codeword<int>(n, q, rng);
//     // print_message("Generated codeword = ", generated_codeword);

//     // Calculate syndrome of generated codeword which will be sent over CC
//     libbase::vector<int> calculated_syndrome(n);
//     cdc->calculate_syndrome(generated_codeword, calculated_syndrome);
//     print_message("Generated syndrome = ", calculated_syndrome);
// // -----------------------------------------------------------------------------
//     /* Still need to figure out why syndrome is of size n not of size m.  
//     */
    
//     /* Modulate codeword */
//     auto mdm = create_modem_gf2();

// #if DEBUG >= 1
// std::cout << "TESTSYNDROMEDECODING: Modem Details: " << mdm->description() << std::endl;
// #endif

//     mdm->set_blocksize(libbase::size_type<libbase::vector>(n));

//     const int M = mdm->num_symbols(); 
//     std::cout << "Size M = " << M << std::endl;

//     // 1. Create and initialize the output vector
//     libbase::vector<libbase::gf2> modulated_codeword(7);

//     // 2. Call modulate with the 3 required arguments:
//     //    (int symbol_count, vector<int>& input, vector<gf2>& output)
//     mdm->modulate(M, generated_codeword, modulated_codeword); 

// #if DEBUG >= 1
//     std::cout << "TESTSYNDROMEDECODING: Modulated codeword: " << std::endl;
//     // Assuming you have a print_message for gf2 or similar
//     print_gf_vector_as_ints(modulated_codeword); 
// #endif

//     /* Transmit codeword through a QSC channel */
//     // Define channel parameters
//     auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

//     // Probability of Substitution, Ps
//     double Ps = 0.1; 
//     qsc_channel->set_parameter(Ps);

//     // Seed the channel
//     qsc_channel->seedfrom(rng);

// #if DEBUG >= 1
//     std::cerr << "TESTSYNDROMEDECODING: Details of QSC Channel: "
//                   << qsc_channel ->description() << std::endl;
//     std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps << std::endl; // To do: ideally you get the channel parameter directly from the channel itself
// #endif

//     libbase::vector<libbase::gf2> received_codeword(7);
//     qsc_channel->transmit(modulated_codeword, received_codeword);

//     std::cout << "Received GF2 codeword:  " << std::endl; 
//     print_gf_vector_as_ints(received_codeword);
  
//     // Initialise probability table.
//     auto prob_table = libbase::vector<libbase::vector<double>>(n);

//     /* Demodulate the received codeword */
//     mdm->demodulate(*qsc_channel, received_codeword, prob_table);

// #if DEBUG >= 1
//     std::cout << "TESTSYNDROMEDECODING: Probability Table: "
//                   << prob_table << std::endl;
// #endif

//     // Seed the codec
//     cdc->seedfrom(rng);

//     /* Decode the demodulated codeword to get the message using the original syndrome*/
//     cdc->init_decoder(prob_table, calculated_syndrome);

//     auto decoded_message_u = libbase::vector<int>(cdc->input_block_size());
//     cdc->decode(decoded_message_u);

// #if DEBUG >= 1
//     std::cout << "TESTSYNDROMEDECODING: Decoded message u: "
//                   << decoded_message_u << std::endl;
// #endif

//     /* Validation of result */
//     libbase::vector<int> validation_codeword(n);
//     cdc->encode(decoded_message_u, validation_codeword);
//     print_message("Validation codeword = ", validation_codeword);

//     // Compute number of errors using hamming distance
//     int num_errors =  libbase::hamming(original_message, decoded_message_u);
    
// #if DEBUG >= 1
//     std::cout << "TESTSYNDROMEDECODING: Validating Result " << std::endl;
//     std::cout << "TESTSYNDROMEDECODING: Encode decoded message: " << validation_codeword << std::endl;
//     std::cout << "TESTSYNDROMEDECODING: Number of errors between original and decoded messages: " << num_errors <<  std::endl;
// #endif

// }

