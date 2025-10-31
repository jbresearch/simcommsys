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


 // Instantiate the AWGN channel object.
auto qsc_channel = std::make_shared<libcomm::qsc<libbase::gf2>>();

// Probability of Substitution, Ps
double Ps = 0.0; 

#if DEBUG >= 1
        std::cerr << "TESTSYNDROMEDECODING: Details of Channel: "
                  << qsc_channel ->description() << std::endl;
        std::cerr << "TESTSYNDROMEDECODING: P_s = " << Ps << std::endl;
#endif

// pass through qsc channel over gf2 without any errors. This outputs the probability table.
// Use doextract from the blind embedder

// decode using do_init_decoder(const array1vdbl_t& ptable, const libbase::vector<int>& syndrome) override;

        // // Instantiate Embedder. 
        // std::shared_ptr<libcomm::block_blind_embedder<double, libbase::vector, double>>
        // embedder; // Embedder

        // // Perform Demodulation to get Probability Table.
        // // codeword_ex1 has no errors, pass directly 
        // embedder->extract(*demodulation_channel,
        //                   codeword_ex1,
        //                   prob_table); 

        // Declare and initialise mapper and modem mapper  
        auto mapper = std::make_shared<libcomm::map_straight<libbase::vector, double>>();
        auto modem = std::make_shared<libcomm::direct_blockmodem<libbase::gf2, libbase::vector, double>>();

        const int alphabet_size = libbase::gf2::elements(); // 2
        const int block_size = codeword_ex1.size();       // N=7

        mapper->set_parameters(alphabet_size, alphabet_size);
        mapper->set_blocksize(libbase::size_type<libbase::vector>(block_size));
        
        modem->set_parameters(alphabet_size, alphabet_size);
        modem->set_blocksize(libbase::size_type<libbase::vector>(block_size));
        
        // Map
        auto mapped_codeword = libbase::vector<int>();
        mapper->transform(codeword_ex1, mapped_codeword);

        // Modulate
        auto modulated_codeword = libbase::vector<libbase::gf2>();
        modem->libcomm::basic_blockmodem<libbase::gf2, libbase::vector, double>::modulate(
        libbase::gf2::elements(),
        mapped_codeword, 
        modulated_codeword
        );

        // Transmit
        auto  corrupted_codeword = libbase::vector<libbase::gf2>();
        qsc_channel->set_parameter(Ps); 
        qsc_channel->transmit(modulated_codeword, corrupted_codeword);

        // Demodulate
        auto demodulated_codeword = libbase::vector<libbase::vector<double>>();
        modem->libcomm::basic_blockmodem<libbase::gf2, libbase::vector, double>::demodulate(
        *qsc_channel, 
        corrupted_codeword, 
        demodulated_codeword
    );

        // Instantiate probabiltiy table
        auto prob_table = libbase::vector<libbase::vector<double>>();
        
        // Inverse Map
        mapper->inverse(demodulated_codeword, prob_table);

#if DEBUG >= 1
        std::cerr << "TESTSYNDROMEDECODING: prob_table = " << prob_table << std::endl;
#endif

        // Decode 
        /*LDPC decoding using the prob_table to get decoded codeword */
        cdc->init_decoder(prob_table, calculated_syndrome_ex1);

        auto decoded_codeword_ex1 = libbase::vector<int>();
        cdc->decode(decoded_codeword_ex1);
        
#if DEBUG >= 1
        std::cerr << "TESTSYNDROMEDECODING: decoded codeword = " << decoded_codeword_ex1 << std::endl;
#endif

BOOST_CHECK_EQUAL(decoded_codeword_ex1.isequalto(codeword_ex1), true);

}

