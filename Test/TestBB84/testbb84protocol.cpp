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
#include "qkd/quantum_state.h"
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

std::pair<bool, bool> get_alice_choice_from_qubit(const libcomm::qubit& q)
{
    // Define the values Alice uses
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double epsilon = 1e-9; // A small tolerance for float comparison

    // Get the internal amplitudes.
    std::complex<double> alpha = q.get_comp_basis_0();
    std::complex<double> beta = q.get_comp_basis_1();

    // Now, compare against the 4 known noiseless states
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
    // We'll throw an error here, as a test should be precise.
    throw std::runtime_error("Unknown qubit state. Not a valid Alice state.");
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
    std::cout << "These vectors are just being printed for testing purpose: " << std::endl;
    print_std_vector("Print bits vector of Alice = ", vector_a);
    print_std_vector("Print bases vector of Alice = ", vector_b);


    // Get measurement value/bit of Alice and the basis vector for a single qubit.
    std::cout << "***** Verification *****" << std::endl;
    std::cout << "Verifying that the bit value and basis value for the first generated qubit is correct: " << std::endl; 

    std::pair<bool, bool> deduced = get_alice_choice_from_qubit(source(0));
    bool deduced_bit = deduced.first;
    bool deduced_basis = deduced.second;

    std::cout << "The deduced bit for quantum state 1 = " << deduced_bit << std::endl;
    std::cout << "The deduced bit for quantum state 1 = " << deduced_bit << std::endl;

    


}
