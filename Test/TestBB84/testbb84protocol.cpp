/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 * \brief Boost unit tests for the BB84 protocol with single polarization which
 * is a DV-QKD protocol.
 */

#define BOOST_TEST_MODULE BB84Test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "serializer_libcomm.h"
#include "experiment/results_collector.h"
#include "experiment/binomial/result_collector/qkd_commsys/dv_qkd_errors_hamming.h"
#include "qkd/qkd_protocol/dvqkd_protocol.h"
#include "qkd/quantum_channel/depolarizing_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "qkd_commsys.h"
#include "source/quantum_bb84_source.h"

#include "codec/ldpc.h"
#include "gf.h"
#include "random.h"
#include "vector.h"

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#   undef DEBUG
#   define DEBUG 1
#endif

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
# Version
1
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
    sys->seedfrom(*rng);

    // Setting and getting CLI Parameter
    const double qber = 0.5; //0.00; //0.06; // QBER, the new CLI parameter in percentage.
    libbase::vector<double> cli;
    cli.init(sys->get_num_params()); // should be 1 when Alice is identity , CLI
                                     // channel parameters
    cli(0) = qber;                   // index 0 -> Bob's QBER to generate noise.
    sys->set_parameters(cli);

    // Print System Parameters of the QKD Commsys Object
    std::cout << "\n" << sys->description() << "\n\n";

#if DEBUG >= 1
    // Verify CLI parameters of Quantum Channel of Bob
    auto channel_parameters = sys->get_parameters();
    std::cout << "TESTBB84:  (CLI parameter of Bob's Quantum Channel) QBER = " << channel_parameters(0) << std::endl;
#endif

    // Create BB84 Source Generator.
    std::stringstream ss_src;
    ss_src << R"SS(
quantum_bb84_source
)SS";

    // Gets the number of coherent states generated for a single frame from the
    // qkd_commsys object.
    int framesize = sys->input_block_size();

#if DEBUG >= 1
    std::cout << "TESTBB84:  Number of qubits (Alice) = " << framesize
              << std::endl;
#endif

    // Build source generator.
    std::shared_ptr<libcomm::source<libcomm::qubit, libbase::vector>> s_ptr;
    ss_src >> s_ptr;
    auto* src = dynamic_cast<libcomm::quantum_bb84_source*>(s_ptr.get());
    BOOST_REQUIRE(src != nullptr);

    // Seed source generator.
    libbase::randgen r;
    r.seed(2602);
    src->seedfrom(r);

    // Generate a sequence of qubits which is the input to the fullcycle method
    // in qkd_commsys.h
    libbase::vector<libcomm::qubit> source =
        src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

    /* Sends source to qkd_commsys by creating a simulator, which calls
     * sys->init() in its constructor.*/
    // Define the template types for the simulator.
    using S = libcomm::qubit;
    using T = bool;
    // using R = libcomm::dv_qkd_errors_hamming;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T>>(
        // Upcast rng from shared_ptr<randgen> to shared_ptr<random>.
        std::static_pointer_cast<libbase::random>(rng),
        s_ptr,
        sys);

    /* Calling fullcycle method from qkd_commsys.h for a single frame */
    auto [key_KA, key_KB] = sys->fullcycle(source);
    std::cout << "TESTBB84: Size of Final Secret Key KA: " << key_KA.size()
              << std::endl;
    std::cout << "TESTBB84: Final Secret Key KA: " << key_KA << std::endl;
    std::cout << "TESTBB84: Final Secret Key KB: " << key_KB << std::endl;


#if DEBUG >= 1
    /* TODO: Still to move this part in a separate test file. */
    // Get vector a of Alice which is the vector of bits.
    std::vector<bool> vector_a(framesize);
    vector_a = src->get_bits();
    // Get vector b of Alice which is the basis vector.
    std::vector<bool> vector_b(framesize);
    vector_b = src->get_bases();
    // In your test case:
    std::cout << "TESTBB84: Verification of basis and bits vectors of Alice: "
              << std::endl;
    std::cout << "TESTBB84:  bits vector of Alice = " << vector_a << std::endl;
    std::cout << "TESTBB84:  bases vector of Alice = " << vector_b << std::endl;
#endif

} // end of Boost test


