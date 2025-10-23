/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 * \brief Boost unit tests for quantum_gaussian_source and cvqkd_protocol
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "serializer_libcomm.h"

#include "experiment/binomial/result_collector/qkd_commsys/cv_qkd_errors_hamming.h"
#include "qkd/qkd_protocol/cvqkd_protocol.h"
#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "qkd_commsys.h"
#include "source/quantum_gaussian_source.h"

#include "codec/ldpc.h"
#include "gf.h"
#include "random.h"
#include "vector.h"

BOOST_AUTO_TEST_CASE(test_qkd_commsys_object_up_until_measurement)
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

    //     std::stringstream cfg;
    //     cfg << R"SS(
    // # Version
    // 1
    // # Frame size (# of quantum states in a frame)
    // 30
    // ## Alice's channel
    // identity_quantum_channel
    // ## Bob's channel
    // gaussian_quantum_channel
    // # Homodyne Detector Efficiency
    // 0.606
    // # Mean of the Gaussian Quantum Channel
    // 0.0
    // # Transmittance T of the Gaussian Quantum Channel
    // 0.302
    // ## Postprocessing protocol
    // cvqkd_protocol
    // # Shot Noise Variance N_0
    // 1
    // # Electric Noise v_el
    // 0.041
    // # Detector Efficiency eta
    // 0.606
    // # Smoothing Parameter
    // 1e-4
    // # Alphabet size
    // 2
    // # Codec
    // ldpc<gf2,double>
    // # Version
    // 5
    // # SPA type (trad|gdl)
    // gdl
    // # Number of iterations
    // 100
    // # Clipping method
    // zero
    // # Value of almostzero
    // 1e-100
    // # Reduce generator matrix to REF? (true|false)
    // 0
    // # Length (n)
    // 15
    // # Dimension (m)
    // 10
    // # Max column weight
    // 2
    // # Max row weight
    // 3
    // # Non-zero values (ones|random|provided)
    // ones
    // # Column weight vector
    // 15
    // 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    // # Row weight vector
    // 10
    // 3 3 3 3 3 3 3 3 3 3
    // # Non zero positions per col
    // 2
    // 1 2
    // 2
    // 2 3
    // 2
    // 3 4
    // 2
    // 4 5
    // 2
    // 1 5
    // 2
    // 1 6
    // 2
    // 2 8
    // 2
    // 3 10
    // 2
    // 4 7
    // 2
    // 5 9
    // 2
    // 6 7
    // 2
    // 7 8
    // 2
    // 8 9
    // 2
    // 9 10
    // 2
    // 6 10
    // # Embedder
    // direct_block_informed_embedder<double,vector,double>
    // sign<double>
    // )SS";

    std::stringstream cfg;
    cfg << R"SS(
# Version
1
# Frame size (# of quantum states in a frame)
14
## Alice's channel
identity_quantum_channel
## Bob's channel
gaussian_quantum_channel
# Homodyne Detector Efficiency
0.606
# Mean of the Gaussian Quantum Channel
0.0
# Transmittance T of the Gaussian Quantum Channel
0.302
## Postprocessing protocol
cvqkd_protocol
# Shot Noise Variance N_0
1
# Electric Noise v_el
0.041
# Detector Efficiency eta
0.606
# Smoothing Parameter
1e-4
# Alphabet size
2
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
# Embedder
direct_block_informed_embedder<double,vector,double>
sign<double>
)SS";

    auto sys = std::make_shared<libcomm::qkd_commsys<libcomm::gaussian_state,
                                                     double,
                                                     libbase::vector>>();

    sys->serialize(cfg);

    // Create rng as a shared_ptr and set the seed.
    auto rng = std::make_shared<libbase::randgen>();
    rng->seed(8);

    /*
    With this seed:
    rng.seed(17);

    and a
    randgen r; // for src and vector s
    r.seed(2602); // for src and  vector s

    I had the following:
    - Generated vector s of Bob = [1 1 1]
    - Generated vector C (after encoding s) = [1       1       1       0       0
    1       0]
    - Generated vector M = -0.324052       -0.172396       0.0642522 0.840933
    -0.233945       -0.0788091      0.305013
    - Generated probability table:
    (Prints probability table from cv-qkdprotocol.cpp:)
    t=0 : 0.000037, 0.361188
    t=1 : 0.396687, 0.059911
    t=2 : 0.392867, 0.335163
    t=3 : 0.201802, 0.000000
    t=4 : 0.388728, 0.004239
    t=5 : 0.005572, 0.000231
    t=6 : 0.049393, 0.112772
    - decoded sequence s_hat = [1 1 1]
    */

    sys->seedfrom(*rng);

    const double VN = 1.041915; // Variance VN, the new CLI parameter.
    // If VA = 18.5, SNR_linear ~ 17.7558

    libbase::vector<double> cli;
    cli.init(sys->get_num_params()); // should be 1 when Alice is identity , CLI
                                     // channel parameters
    cli(0) = VN; // index 0 -> Bob's Variance VN to generate noise.

    sys->set_parameters(cli);

    // 4) Print System Parameters of the QKD Commsys Object
    std::cout << "\n" << sys->description() << "\n\n";

    // 5) Verify CLI parameters of Quantum Channel of Bob
    auto back = sys->get_parameters();

    std::cout << "TESTGAUSSIANCVQKD:  (CLI parameter of Bob's Quantum Channel) "
                 "Variance VN = "
              << VN << std::endl;

    // 6) Create Gaussian Quantum Source
    std::stringstream ss_src;
    ss_src << R"SS(
quantum_gaussian_source
# Mean of Q_Mean
0.0
# Stddev of Q_Mean
4.30116
# Mean of P_Mean
0.0
# Stddev of P_Mean
4.30116
# Stddev of Q
1.0
# Stddev of P
1.0
)SS";

    // Build source using the same pattern as gaussian_quantum_channel
    // std::shared_ptr<libcomm::source<libcomm::gaussian_state,
    // libbase::vector>>
    //     s_ptr;

    std::shared_ptr<libcomm::source<libcomm::gaussian_state>> s_ptr;

    ss_src >> s_ptr;
    auto* src = dynamic_cast<libcomm::quantum_gaussian_source*>(s_ptr.get());
    BOOST_REQUIRE(src != nullptr);

    libbase::randgen r;
    r.seed(2602);
    // r.seed(31);
    src->seedfrom(r);

    // Gets the number of coherent states generated for a single frame from the
    // qkd_commsys object.
    int framesize = sys->input_block_size();
    std::cout
        << "TESTGAUSSIANCVQKD:  Number of generated coherent states (Alice) = "
        << framesize << std::endl;

    // Generates a sequence of coherent states which is the input to the
    // fullcycle method in qkd_commsys.h
    libbase::vector<libcomm::gaussian_state> source =
        src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

    /* Sends source to qkd_commsys by creating a simulator, which calls
     * sys->init() in its constructor.*/
    // Define the template types for the simulator.
    using S = libcomm::gaussian_state;
    using T = double;
    using R = libcomm::cv_qkd_errors_hamming;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T, R>>(
        // Upcast rng from shared_ptr<randgen> to shared_ptr<random>.
        std::static_pointer_cast<libbase::random>(rng),
        s_ptr,
        sys);

    // At this point, sys->init() has been called and the system including the
    // protocol is fully initialised.

    // Initialise final_key
    libbase::vector<bool> final_key;

    /* Calling fullcylce method from qkd_commsys.h for a single frame*/
    auto [key_KA, key_KB] = sys->fullcycle(source);

    std::cout << "TESTGAUSSIANCVQKD:  Size of Final Secret Key KA: "
              << key_KA.size() << std::endl;
}
