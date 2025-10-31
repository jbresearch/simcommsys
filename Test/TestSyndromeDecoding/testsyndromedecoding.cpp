/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 * \brief Boost unit tests for the BB84 protocol with single polarization which
 * is a DV-QKD protocol.
 */

#define BOOST_TEST_MODULE BB84Test
#include <boost/test/included/unit_test.hpp>

#include "serializer_libcomm.h"
#include "codec/ldpc.h"
#include "gf.h"
#include "matrix.h"
#include "channel/qsc.h"
#include "mapper/map_straight.h"
#include "modem/direct_blockmodem.h"
#include "random.h"
// #include "blind_embedder/direct_block_blind_embedder.h"

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#   undef DEBUG
#   define DEBUG 1
#endif

// For Boost Test 1 
// Step 1: Serialize a codec of LDPC type with the respective matrix I will use.
// Step 2: Calculate the syndrome 
// Step 3: Compare calculated syndrome with original one (that Alice should know beforehand). 

/* 
    Examples taken from slides of Victor.

    Example 1: codeword without errors: [1 0 1 1 1 0 0]
    Thus syndrome should be an all zero syndrome: s = [0 0 0 0 0 0 0]

    Example 2: codeword with errors: [1 1 1 1 0 0 0]
    The respective syndrome should be: s = [1 0 0 1 1 1 0] 
*/ 

BOOST_AUTO_TEST_CASE(test_syndrome_decoding_without_errors)
{
    // Make sure we instantiate everything
    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << "Boost Test 1: Testing Syndrome Decoding for GF2 without Errors" << std::endl;

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

// LDPC codec 
auto cdc = std::make_shared<
        libcomm::ldpc<libbase::gf2, double>>();

cdc->serialize(cfg);


#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Codec Details: " << cdc->description()
              << std::endl;
#endif

/*    Example 1: codeword without errors: [1 0 1 1 1 0 0]
    Thus syndrome should be an all zero syndrome: s = [0 0 0 0 0 0 0]
    Example is taken from: "Part 5 - Low Density Parity Check Codes.pdf", written by Victor Buttigieg
*/

auto codeword_ex1 = libbase::vector<int>(std::vector<int>{1, 0, 1, 1, 1, 0, 0});
const auto expected_syndrome_ex1 = libbase::vector<int>(std::vector<int>{0, 0, 0, 0, 0, 0, 0});

libbase::vector<int> calculated_syndrome_ex1;
calculated_syndrome_ex1.init(7);

// calculate the syndrome 
cdc->calculate_syndrome(codeword_ex1, calculated_syndrome_ex1);  

#if DEBUG >= 1
    std::cout << "TESTSYNDROMEDECODING: Example 1 - Without Errors" << std::endl;
    std::cout << "codeword: " << codeword_ex1 << std::endl;
    std::cout << "expected syndrome: " << expected_syndrome_ex1 << std::endl;
    std::cout << "calculated syndrome: " << calculated_syndrome_ex1  << std::endl;
#endif

BOOST_CHECK_EQUAL(calculated_syndrome_ex1.isequalto(expected_syndrome_ex1), true);

// Define channel parameters
auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

// Probability of Substitution, Ps
double Ps = 0.0; 
qsc_channel->set_parameter(Ps);

// Create rng as a shared_ptr and set the seed.
auto rng = std::make_shared<libbase::randgen>();
rng->seed(7);
// Seed the channel
qsc_channel->seedfrom(*rng);

//     // Seed source generator.
//     libbase::randgen r;
//     r.seed(2602);
//     src->seedfrom(r);

#if DEBUG >= 1
        std::cerr << "TESTSYNDROMEDECODING: Details of Channel: "
                  << qsc_channel ->description() << std::endl;
        std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps << std::endl; // To do: ideally you get the channel parameter directly from the channel itself
#endif

        // // Instantiate Embedder. 
        // std::shared_ptr<libcomm::block_blind_embedder<double, libbase::vector, double>>
        // embedder; // Embedder

        // // Perform Demodulation to get Probability Table.
        // // codeword_ex1 has no errors, pass directly 
        // embedder->extract(*demodulation_channel,
        //                   codeword_ex1,
        //                   prob_table); 

        // Transmit codeword through a qsc channel over gf2 -- to resolve a bad alloc here. 
        // const libbase::vector<libbase::gf2> gf2_codeword(codeword_ex1);

        libbase::vector<libbase::gf2> gf2_codeword;
        gf2_codeword.init(codeword_ex1.size());

        for (int i = 0; i < codeword_ex1.size(); ++i)
        {
                gf2_codeword(i) = libbase::gf2(codeword_ex1(i));
        }
        std::cout << "GF2 codeword example 1 = " << gf2_codeword(0) << std::endl;

        // libbase::vector<libbase::gf2> corrupted_codeword;
        // corrupted_codeword.init(7);
        // qsc_channel->transmit(gf2_codeword, corrupted_codeword);

// #if DEBUG >= 1
//         std::cerr << "TESTSYNDROMEDECODING: corrupted codeword = " << corrupted_codeword << std::endl;
// #endif

//         // Hardcoded Probability Table
//         // Codeword assumed to be received: 1, 0, 1, 1, 1, 0, 0
//         // Flip probability Ps =  0
//         // ** FIX 1: Use default constructor and .init() to avoid bad_alloc **
//         auto prob_table = libbase::vector<libbase::vector<double>>(7);
//         //                                     Probability bit is    0 ,  1
//         prob_table(0) = libbase::vector<double>(std::vector<double>{0.1, 0.9}); // 1
//         prob_table(1) = libbase::vector<double>(std::vector<double>{0.9, 0.1}); // 0
//         prob_table(2) = libbase::vector<double>(std::vector<double>{0.1, 0.9}); // 1
//         prob_table(3) = libbase::vector<double>(std::vector<double>{0.1, 0.9}); // 1
//         prob_table(4) = libbase::vector<double>(std::vector<double>{0.1, 0.9}); // 1
//         prob_table(5) = libbase::vector<double>(std::vector<double>{0.9, 0.1}); // 0
//         prob_table(6) = libbase::vector<double>(std::vector<double>{0.9, 0.1}); // 0

// #if DEBUG >= 1
//         std::cerr << "TESTSYNDROMEDECODING: prob_table = " << prob_table << std::endl;
// #endif

//         // Decode 
//         /*LDPC decoding using the prob_table to get decoded codeword */
//         cdc->init_decoder(prob_table, calculated_syndrome_ex1);

//         auto decoded_codeword_ex1 = libbase::vector<int>();
//         cdc->decode(decoded_codeword_ex1);
        
// #if DEBUG >= 1
//         std::cerr << "TESTSYNDROMEDECODING: decoded codeword = " << decoded_codeword_ex1 << std::endl;
// #endif

// BOOST_CHECK_EQUAL(decoded_codeword_ex1.isequalto(codeword_ex1), true);

}

