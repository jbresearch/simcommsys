/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 * \brief Boost unit tests for the BB84 protocol with single polarization which is a DV-QKD protocol.
 */

#define BOOST_TEST_MODULE BB84Test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "serializer_libcomm.h"

#include "qkd/qkd_protocol/dvqkd_protocol.h"
#include "qkd/quantum_channel/depolarizing_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "qkd_commsys.h"
#include "source/quantum_bb84_source.h"
#include "experiment/binomial/result_collector/qkd_commsys/dv_qkd_errors_hamming.h"

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

    std::cout << "Base classes:" << std::endl;
    for (auto& s : libbase::serializer::get_base_classes())
        std::cout << " - " << s << std::endl;
    std::cout << "Derived classes for quantum_channel:" << std::endl;
    for (auto& s : libbase::serializer::get_derived_classes("quantum_channel"))
        std::cout << " - " << s << std::endl;


std::stringstream cfg;
    cfg << R"SS(
# Version
1
# Frame size (# of quantum states in a frame)
14
## Alice's channel
identity_quantum_channel
## Bob's channel
depolarizing_quantum_channel
## Postprocessing protocol
dvqkd_protocol
# Codec
ldpc<gf2,double>
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

    auto sys = std::make_shared<
    libcomm::qkd_commsys<libcomm::qubit, bool, libbase::vector>>();

    sys->serialize(cfg);

    // Create rng as a shared_ptr and set the seed.
    auto rng = std::make_shared<libbase::randgen>();
    rng->seed(7);


    // Create BB84 Source Generator.
    std::stringstream ss_src;
    ss_src << R"SS(
quantum_bb84_source
)SS";

    // Gets the number of coherent states generated for a single frame from the
    // qkd_commsys object.
    int framesize = sys->input_block_size();
    std::cout
        << "TESTGAUSSIANCVQKD:  Number of generated coherent states (Alice) = "
        << framesize << std::endl;


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

    /* Sends source to qkd_commsys by creating a simulator, which calls sys->init() in its constructor.*/
    // Define the template types for the simulator.
    using S = libcomm::qubit;
    using T = bool;
    using R = libcomm::dv_qkd_errors_hamming;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T, R>>(

        // Upcast rng from shared_ptr<randgen> to shared_ptr<random>.
        std::static_pointer_cast<libbase::random>(rng),

        s_ptr,

        sys
    );


    // Get vector a of Alice which is the vector of bits.
    std::vector<bool> vector_a(framesize);

    vector_a = src->get_bits();

    // Get vector b of Alice which is the basis vector.
    std::vector<bool> vector_b(framesize);

    vector_b = src->get_bases();

    // In your test case:
    std::cout << "Verification of basis and bits vectors of Alice: " << std::endl;
    print_std_vector("Print bits vector of Alice = ", vector_a);
    print_std_vector("Print bases vector of Alice = ", vector_b);
}
