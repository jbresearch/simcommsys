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

#include "source/quantum_bb84_source.h"
#include "qkd_commsys.h"

// #include "codec/ldpc.h"
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

// Helper function to print a vector
template<typename T>
void print_std_vector(const std::string& title, const std::vector<T>& vec) {
    std::cout << title;
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}


BOOST_AUTO_TEST_CASE(test_bb84_protocol)
{
    // Make sure we instantiate everything
    const libcomm::serializer_libcomm my_serializer_libcomm;

    std::cout << "\n*****Boost Test Case *****\n";

    // Create BB84 Source Generator.
    std::stringstream ss_src;
    ss_src << R"SS(
quantum_bb84_source
)SS";

    // Number of qubits generated for a single frame
    int framesize = 10;

    // Build source.
    std::shared_ptr<libcomm::source<libcomm::qubit, libbase::vector>>
        s_ptr;
    ss_src >> s_ptr;
    auto* src = dynamic_cast<libcomm::quantum_bb84_source*>(s_ptr.get());
    BOOST_REQUIRE(src != nullptr);

    libbase::randgen r;
    r.seed(2602);
    src->seedfrom(r);

    // Generate a sequence of qubits which is the input to the fullcycle method in qkd_commsys.h
    libbase::vector<libcomm::qubit> source =
        src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

    // Get vector a of Alice which is the vector of bits.
    std::vector<bool> vector_a(framesize);

    vector_a = src->get_bits();

    // Get vector b of Alice which is the basis vector.
    std::vector<bool> vector_b(framesize);

    vector_b = src->get_bases();

    // In your test case:
    print_std_vector("Print bits vector of Alice = ", vector_a);
    print_std_vector("Print bases vector of Alice = ", vector_b);

}
