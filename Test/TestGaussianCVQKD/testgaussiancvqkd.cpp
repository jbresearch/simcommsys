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

#include "experiment/results_collector.h"
#include "experiment/binomial/result_collector/qkd_commsys/qkd_errors_hamming.h"
#include "qkd/qkd_protocol/cvqkd_protocol.h"
#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "qkd_commsys.h"
#include "source/quantum_gaussian_source.h"

#include "codec/ldpc.h"
#include "gf.h"
#include "random.h"
#include "vector.h"

BOOST_AUTO_TEST_CASE(test_fullcycle_for_single_vn)
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

//     // Test 1 : n = 7, k = 3, m =4 
//     std::stringstream cfg;
//     cfg << R"SS(
// # Version
// 1
// # Frame size (# of quantum states in a frame)
// 100000
// ## Alice's channel
// identity_quantum_channel
// ## Bob's channel
// gaussian_quantum_channel
// # Mean of the Gaussian Quantum Channel
// 0.0
// # Fading Coefficient alpha
// 0.34641
// ## Postprocessing protocol
// cvqkd_protocol
// # Version
// 1
// # N_PE
// 99993
// # VA, VN, alpha from parameter estimation?
// 1
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
// 50
// # Clipping method
// zero
// # Value of almostzero
// 1e-100
// # Reduce generator matrix to REF? (true|false)
// 1
// # Length (n)
// 7
// # Dimension (m)
// 7
// # Max column weight
// 3
// # Max row weight
// 3
// # Non-zero values (ones|random|provided)
// ones
// # Column weight vector
// 7
// 3 3 3 3 3 3 3
// # Row weight vector
// 7
// 3 3 3 3 3 3 3
// # Non zero positions per col
// 3
// 1 5 7
// 3
// 1 2 6
// 3
// 2 3 7
// 3
// 1 3 4
// 3
// 2 4 5
// 3
// 3 5 6
// 3
// 4 6 7
// # Embedder
// direct_block_informed_embedder<double,vector,double>
// sign<double>
// )SS";

     // Test 2 : n = 15, k = , m =  
    std::stringstream cfg;
    cfg << R"SS(
# Version
1
# Frame size (# of quantum states in a frame)
30
## Alice's channel
identity_quantum_channel
## Bob's channel
gaussian_quantum_channel
# Mean of the Gaussian Quantum Channel
0.0
# Fading Coefficient alpha
0.34641
## Postprocessing protocol
cvqkd_protocol
# Version
1
# N_PE
15
# VA, VN, alpha from parameter estimation?
1
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
100
# Clipping method
zero
# Value of almostzero
1e-100
# Reduce generator matrix to REF? (true|false)
0
# Length (n)
15
# Dimension (m)
10
# Max column weight
2
# Max row weight
3
# Non-zero values (ones|random|provided)
ones
# Column weight vector
15
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# Row weight vector
10
3 3 3 3 3 3 3 3 3 3
# Non zero positions per col
2
1 2
2
2 3
2
3 4
2
4 5
2
1 5
2
1 6
2
2 8
2
3 10
2
4 7
2
5 9
2
6 7
2
7 8
2
8 9
2
9 10
2
6 10
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

    // const double VN = 0.0004302606584964948; //1.041915; // Variance VN, the new CLI parameter.
    // If VA = 18.5, SNR_linear ~ 17.7558

    /* Test Cases using the data from the csv file: qudice_atmospheric_results1 
    // SNR_linear = 2.80E-05, elevation angle = 0.344387634 degrees
    // const double VN = 77072.88; 

    // SNR_linear = 1.335929371, SNR_dB = 1.25783498,  elevation angle = 40.34405295 degrees
    // const double VN = 1.61685; 
    */ 
    // SNR_linear = 3.104172843 , SNR_dB = 4.919458951,  elevation angle = 89.66604521 degrees
    const double VN = 0.6958369;

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

// Test case for optical fiber with the below parameters: 
// 1.041915; // Variance VN, the new CLI parameter.
// If VA = 18.5, SNR_linear ~ 17.7558 
// 6) Create Gaussian Quantum Source
//     std::stringstream ss_src;
//     ss_src << R"SS(
// quantum_gaussian_source
// # Mean of Q_Mean
// 0.0
// # Stddev of Q_Mean
// 4.30116
// # Mean of P_Mean
// 0.0
// # Stddev of P_Mean
// 4.30116
// # Stddev of Q
// 1.0
// # Stddev of P
// 1.0
// )SS";


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
    // using R = libcomm::qkd_errors_hamming;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T>>(
        // Upcast rng from shared_ptr<randgen> to shared_ptr<random>.
        std::static_pointer_cast<libbase::random>(rng),
        s_ptr,
        sys);

    // At this point, sys->init() has been called and the system including the
    // protocol is fully initialised.

    // Initialise final_key
    libbase::vector<bool> final_key;

    /* Calling fullcycle method from qkd_commsys.h for a single frame*/
    auto [key_KA, key_KB] = sys->fullcycle(source);

    std::cout << "TESTGAUSSIANCVQKD:  Size of Final Secret Key KA: "
              << key_KA.size() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_cvqkd_fullcyclecvqkdresults_single_vn, *boost::unit_test::disabled())
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
100000
## Alice's channel
identity_quantum_channel
## Bob's channel
gaussian_quantum_channel
# Mean of the Gaussian Quantum Channel
0.0
# Fading Coefficient alpha
0.34641
## Postprocessing protocol
cvqkd_protocol
# Version
1
# N_PE
99993
# VA, VN, alpha from parameter estimation?
1
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

    // const double VN = 0.0004302606584964948; //1.041915; // Variance VN, the new CLI parameter.
    // If VA = 18.5, SNR_linear ~ 17.7558

    /* Test Cases using the data from the csv file: qudice_atmospheric_results1 
    // SNR_linear = 2.80E-05, elevation angle = 0.344387634 degrees
    // const double VN = 77072.88; 

    // SNR_linear = 1.335929371, SNR_dB = 1.25783498,  elevation angle = 40.34405295 degrees
    // const double VN = 1.61685; 
    */ 
    // SNR_linear = 3.104172843 , SNR_dB = 4.919458951,  elevation angle = 89.66604521 degrees
    const double VN = 0.6958369;

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

// Modulation variance VA = 18.5
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
    // using R = libcomm::qkd_errors_hamming;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T>>(
        // Upcast rng from shared_ptr<randgen> to shared_ptr<random>.
        std::static_pointer_cast<libbase::random>(rng),
        s_ptr,
        sys);

    // At this point, sys->init() has been called and the system including the
    // protocol is fully initialised.

    // Initialise final_key
    libbase::vector<bool> final_key;

    /* Calling fullcycle method from qkd_commsys.h for a single frame*/
    auto [MI_Check, I_AB, chi_BE, VA_hat, VN_result, VN_hat, alpha_hat, len_secret_key] = sys->fullcyclecvqkdresults(source);

    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Variance VN: " 
              << VN_result << std::endl; // VN already existed as a variable so I had to rename this
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Variance VN_hat: " 
              << VN_hat << std::endl;
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Variance VA_hat: " 
              << VA_hat << std::endl;
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Variance alpha_hat: " 
              << alpha_hat << std::endl;
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Mutual Information Check: " 
              << MI_Check << std::endl;
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Mutual Information I_AB: " 
              << I_AB << std::endl;
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Holevo Bound: " 
              << chi_BE << std::endl;
    std::cout << "TESTGAUSSIANCVQKD_RESULTS:  Length l of secret key: " 
              << len_secret_key << std::endl;
}

// Helper function to split a CSV line
std::vector<std::string> split_csv_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(line);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

BOOST_AUTO_TEST_CASE(test_cvqkd_batch_processing_from_csv, *boost::unit_test::disabled())
{
    std::cout << "\n***** Starting Batch CSV Processing *****\n";

    // Define File Paths
    // Input File (Read from Test_Data)
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/fog_cirrus_alpha_0_34641.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/clear_thin_cirrus_alpha_0_34641.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/fog_thin_cirrus_alpha_0_34641.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/clear_cirrus_alpha_0_34641.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/snow_thin_cirrus_alpha_0_34641.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/rain_cirrus_alpha_0_34641.csv";
    const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/snow_cirrus_alpha_0_34641.csv"; 
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/rain_thin_cirrus_alpha_0_34641.csv";


    // Output File (Write to Results folder)
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/fog_cirrus_alpha_0_34641_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/clear_thin_cirrus_alpha_0_34641_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/fog_thin_cirrus_alpha_0_34641_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/clear_cirrus_alpha_0_34641_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/snow_thin_cirrus_alpha_0_34641_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/rain_cirrus_alpha_0_34641_results.csv";
    const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/snow_cirrus_alpha_0_34641_results.csv"; 
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/rain_thin_cirrus_alpha_0_34641_results.csv";

    
    
    std::stringstream cfg;

    /* QKD Commsys Serialisation */
    cfg << R"SS(
# Version
1
# Frame size (# of quantum states in a frame)
100000
## Alice's channel
identity_quantum_channel
## Bob's channel
gaussian_quantum_channel
# Mean of the Gaussian Quantum Channel
0.0
# Fading Coefficient alpha
0.34641
## Postprocessing protocol
cvqkd_protocol
# Version
1
# N_PE
99993
# VA, VN, alpha from parameter estimation?
1
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

    // Setup RNG and Source
    auto rng = std::make_shared<libbase::randgen>();
    rng->seed(8);
    sys->seedfrom(*rng);

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

    std::shared_ptr<libcomm::source<libcomm::gaussian_state>> s_ptr;
    ss_src >> s_ptr;
    auto* src = dynamic_cast<libcomm::quantum_gaussian_source*>(s_ptr.get());

    libbase::randgen r_src;

    // Initialise qkd_simulator
    using S = libcomm::gaussian_state;
    using T = double;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T>>(
        std::static_pointer_cast<libbase::random>(rng),
        s_ptr,
        sys);
    
    /* Read original CSV file */
    std::ifstream file_in(input_csv_filename);
    if (!file_in.is_open()) {
        BOOST_FAIL("Could not open input CSV file: " + input_csv_filename);
    }

    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    
    // Read all lines
    while (std::getline(file_in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // Handle Windows line endings
        csv_data.push_back(split_csv_line(line, ','));
    }
    file_in.close();

    /* Process Rows and Run Simulation */
    
    const int IDX_ALPHA_HAT = 7;
    const int IDX_VA_HAT = 10;
    const int IDX_VN = 11;
    const int IDX_VN_HAT = 12;
    const int IDX_I_AB = 13;
    const int IDX_CHI_BE = 14;
    const int IDX_MI_CHECK = 15;

    std::cout << "Processing " << csv_data.size() - 1 << " rows..." << std::endl;

    // Start from i = 1 to skip header
    for (size_t i = 1; i < csv_data.size(); ++i) {
        try {

            // Reset the seed to ensure Alice generates the exact same sequence 
            // every time. This isolates the effect of changing VN.
            r_src.seed(2602);
            src->seedfrom(r_src);

            // Generate unmeasured sequence of states
            int framesize = sys->input_block_size();
            // std::cout << "Generating source sequence of size " << framesize << "..." << std::endl;
            libbase::vector<libcomm::gaussian_state> source = 
            src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

            // Get VN from CSV
            std::string vn_str = csv_data[i][IDX_VN];
            if (vn_str.empty()) continue; // Skip empty lines
            
            double current_vn = std::stod(vn_str);

            // Update channel parameter
            libbase::vector<double> cli;
            cli.init(sys->get_num_params());
            cli(0) = current_vn; 
            sys->set_parameters(cli);

            // Run Simulation
            auto [MI_Check, I_AB, chi_BE, VA_hat, VN_res, VN_hat, alpha_hat, len_key] = 
                sys->fullcyclecvqkdresults(source);

            // Update CSV Data in memory
            // Ensure vector is large enough (handle trailing empty commas)
            if (csv_data[i].size() <= IDX_MI_CHECK) {
                csv_data[i].resize(IDX_MI_CHECK + 1);
            }

            csv_data[i][IDX_ALPHA_HAT] = std::to_string(alpha_hat);
            csv_data[i][IDX_VA_HAT] = std::to_string(VA_hat);
            csv_data[i][IDX_VN_HAT] = std::to_string(VN_hat);
            csv_data[i][IDX_I_AB] = std::to_string(I_AB);
            csv_data[i][IDX_CHI_BE] = std::to_string(chi_BE);
            csv_data[i][IDX_MI_CHECK] = std::to_string(MI_Check);

            std::cout << "Row " << i << ": VN =" << current_vn 
                                << " -> alpha_hat =" << alpha_hat 
                                << " -> I_AB =" << I_AB << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error processing row " << i << ": " << e.what() << std::endl;
        }
    }

    /* SAVE TO OUTPUT CSV FILE */
  
    std::ofstream file_out(output_csv_filename); // Writes to new file
    if (!file_out.is_open()) {
        BOOST_FAIL("Could not open output CSV file for writing: " + output_csv_filename);
    }

    for (const auto& row : csv_data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file_out << row[j];
            if (j < row.size() - 1) file_out << ",";
        }
        file_out << "\n";
    }
    file_out.close();

    std::cout << "Batch processing complete. Results saved to " << output_csv_filename << std::endl;
}

BOOST_AUTO_TEST_CASE(test_cvqkd_batch_processing_from_csv_signal_gated,  *boost::unit_test::disabled())
{
    std::cout << "\n***** Starting Batch CSV Processing *****\n";

    // Define File Paths
    // Input File (Read from Test_Data)
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/fog_thin_cirrus_rep_rate_200_snr_window_03_ns.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/snow_cirrus_rep_rate_200_snr_window_03_ns.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/snow_thin_cirrus_rep_rate_200_snr_window_03_ns.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/clear_rep_rate_200_snr_window_03_ns.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/rain_thin_cirrus_rep_rate_200_snr_window_03_ns.csv";
    // const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/rain_cirrus_rep_rate_200_snr_window_03_ns.csv";
    const std::string input_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Test_Data/Test_Data_Signal_Gated/fog_cirrus_rep_rate_200_snr_window_03_ns.csv";


    // Output File (Write to Results folder)
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/fog_thin_cirrus_rep_rate_200_snr_window_03_ns_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/snow_cirrus_rep_rate_200_snr_window_03_ns_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/snow_thin_cirrus_rep_rate_200_snr_window_03_ns_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/clear_rep_rate_200_snr_window_03_ns_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/rain_thin_cirrus_rep_rate_200_snr_window_03_ns_results.csv";
    // const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/rain_cirrus_rep_rate_200_snr_window_03_ns_results.csv";
    const std::string output_csv_filename = "/home/aaron7/git_projects/simcommsys/Test/TestGaussianCVQKD/Results/Results_signal_gated/fog_thin_cirrus_rep_rate_200_snr_window_03_ns_results.csv";


    std::stringstream cfg;
    /* QKD Commsys Serialisation */
    cfg << R"SS(
# Version
1
# Frame size (# of quantum states in a frame)
100000
## Alice's channel
identity_quantum_channel
## Bob's channel
gaussian_quantum_channel
# Mean of the Gaussian Quantum Channel
0.0
# Fading Coefficient alpha
0.34641
## Postprocessing protocol
cvqkd_protocol
# Version
1
# N_PE
99993
# VA, VN, alpha from parameter estimation?
1
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

    // Setup RNG and Source
    auto rng = std::make_shared<libbase::randgen>();
    rng->seed(8);
    sys->seedfrom(*rng);

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

    std::shared_ptr<libcomm::source<libcomm::gaussian_state>> s_ptr;
    ss_src >> s_ptr;
    auto* src = dynamic_cast<libcomm::quantum_gaussian_source*>(s_ptr.get());

    libbase::randgen r_src;

    // Initialise qkd_simulator
    using S = libcomm::gaussian_state;
    using T = double;

    auto sim = std::make_shared<libcomm::qkd_commsys_simulator<S, T>>(
        std::static_pointer_cast<libbase::random>(rng),
        s_ptr,
        sys);
    
    /* Read original CSV file */
    std::ifstream file_in(input_csv_filename);
    if (!file_in.is_open()) {
        BOOST_FAIL("Could not open input CSV file: " + input_csv_filename);
    }

    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    
    // Read all lines
    while (std::getline(file_in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // Handle Windows line endings
        csv_data.push_back(split_csv_line(line, ','));
    }
    file_in.close();

    /* Process Rows and Run Simulation */
    
    const int IDX_ALPHA_HAT = 7;
    const int IDX_VA_HAT = 10;
    const int IDX_VN = 11;
    const int IDX_VN_HAT = 12;
    const int IDX_I_AB = 13;
    const int IDX_CHI_BE = 14;
    const int IDX_MI_CHECK = 15;

    std::cout << "Processing " << csv_data.size() - 1 << " rows..." << std::endl;

    // Start from i = 1 to skip header
    for (size_t i = 1; i < csv_data.size(); ++i) {
        try {

            // Reset the seed to ensure Alice generates the exact same sequence 
            // every time. This isolates the effect of changing VN.
            r_src.seed(2602);
            src->seedfrom(r_src);

            // Generate unmeasured sequence of states
            int framesize = sys->input_block_size();
            // std::cout << "Generating source sequence of size " << framesize << "..." << std::endl;
            libbase::vector<libcomm::gaussian_state> source = 
            src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

            // Get VN from CSV
            std::string vn_str = csv_data[i][IDX_VN];
            if (vn_str.empty()) continue; // Skip empty lines
            
            double current_vn = std::stod(vn_str);

            // Update channel parameter
            libbase::vector<double> cli;
            cli.init(sys->get_num_params());
            cli(0) = current_vn; 
            sys->set_parameters(cli);

            // Run Simulation
            auto [MI_Check, I_AB, chi_BE, VA_hat, VN_res, VN_hat, alpha_hat, len_key] = 
                sys->fullcyclecvqkdresults(source);

            // Update CSV Data in memory
            // Ensure vector is large enough (handle trailing empty commas)
            if (csv_data[i].size() <= IDX_MI_CHECK) {
                csv_data[i].resize(IDX_MI_CHECK + 1);
            }

            csv_data[i][IDX_ALPHA_HAT] = std::to_string(alpha_hat);
            csv_data[i][IDX_VA_HAT] = std::to_string(VA_hat);
            csv_data[i][IDX_VN_HAT] = std::to_string(VN_hat);
            csv_data[i][IDX_I_AB] = std::to_string(I_AB);
            csv_data[i][IDX_CHI_BE] = std::to_string(chi_BE);
            csv_data[i][IDX_MI_CHECK] = std::to_string(MI_Check);

            std::cout << "Row " << i << ": VN =" << current_vn 
                                << " -> alpha_hat =" << alpha_hat 
                                << " -> I_AB =" << I_AB << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error processing row " << i << ": " << e.what() << std::endl;
        }
    }

    /* SAVE TO OUTPUT CSV FILE */
  
    std::ofstream file_out(output_csv_filename); // Writes to new file
    if (!file_out.is_open()) {
        BOOST_FAIL("Could not open output CSV file for writing: " + output_csv_filename);
    }

    for (const auto& row : csv_data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file_out << row[j];
            if (j < row.size() - 1) file_out << ",";
        }
        file_out << "\n";
    }
    file_out.close();

    std::cout << "Batch processing complete. Results saved to " << output_csv_filename << std::endl;
}