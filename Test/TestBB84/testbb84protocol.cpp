/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 * \brief Boost unit tests for the BB84 protocol which is a DV-QKD protocol.
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "serializer_libcomm.h"

#include "qkd_commsys.h"

#include "codec/ldpc.h"
#include "gf.h"
#include "random.h"
#include "vector.h"

template <typename T>
void
print_vector(const std::string& title, const libbase::vector<T>& vec)
{
    std::cout << "\n" << title << std::endl;
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec(i) << "\t";
    }
    std::cout << std::endl;
}

class bit_vector_generator
{
public:
    libbase::vector<bool> generate_vector(int k, libbase::random& r)
    {
        libbase::vector<bool> s(k);
        for (int i = 0; i < k; ++i) {
            s(i) = (r.ival(2) != 0);
        }
        return s;
    }
} sgen;



BOOST_AUTO_TEST_CASE(test_bb84_protocol)
{
    // Make sure we instantiate everything
    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << "\n*****Boost Test Case *****\n";

    // PRNG for vector a
    libbase::randgen rng;

    // Set seed for qkd_commsys object
    rng.seed(17);

    // Number of bits that Alice generates for bit string a.
    int n = 100;

    // Bit string a will be generated in the qkd_commsys_simulator.
    libbase::vector<bool> bit_string_a(n);
    bit_string_a = sgen.generate_vector(n, rng);

    std::cout << "Size of bit_string a: " << bit_string_a.size() << std::endl;
    print_vector("Generated bit string a by Alice: ", bit_string_a);
}
