/*!
 * \file
 * \brief Boost unit tests for quantum_gaussian_source and cvqkd_protocol
 *
 * Copyright (c) 2025 Aaron Abela
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>


#include "source/quantum_gaussian_source.h"
#include "qkd/observable/position_observable.h"
#include "qkd/observable/momentum_observable.h"
#include "qkd/qkd_protocol/cvqkd_protocol.h"
#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "truerand.h"

#include "qkd_commsys.h"
#include "qkd/quantum_state.h"
#include "qkd/qkd_protocol.h"
#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "qkd_commsys.h"
#include "source/quantum_gaussian_source.h"
#include "random.h"
#include "vector.h"

#include <memory>
#include <vector>

#include <iostream>
#include <cmath>

using namespace libcomm;
using namespace libbase;



// BOOST_AUTO_TEST_CASE(test_get_va_function_from_source)
// {
//    std::cout << "\n Boost Test Case: Testing getter of VA from quantum gaussian source" << std::endl;
//    // 1. Generate GM coherent states
//    // Declaring the source without using serialization
//    double variance_VA = 18.5; // modulation variance of Alice
//    double std_dev_VA = std::sqrt(variance_VA);
//    double q_stddev = 1.0;
//    double p_stddev = 1.0;

//    quantum_gaussian_source source(0.0, std_dev_VA, 0.0, std_dev_VA, q_stddev, p_stddev);

//    // ----- Case 1: Used for non serialized
//    randgen r;
//    r.seed(7896);
//    source.seedfrom(r);

//    // double VA = source.get_VA();
//    double VA = qkd_fullcycletest(source);
//    std::cout<< "Modulation Variance VA = " << VA << std::endl;
// }

vector<gaussian_state>
generate_states(libcomm::source<gaussian_state, libbase::vector>& source, int framesize)
{
    vector<gaussian_state> seq =
        source.generate_sequence(size_type<vector>(framesize));
    std::cout << "Number of Generated Coherent States: " << framesize << std::endl;
    return seq;
}

BOOST_AUTO_TEST_CASE(create_measurement_vectors)
{
     std::cout << "\n Boost Test Case: Creating the measurement vectors of Alice and Bob" << std::endl;
   // 1. Generate GM coherent states
   // Declaring the source without using serialization
   // double variance_VA = 18.5; // modulation variance of Alice
   // double std_dev_VA = std::sqrt(variance_VA);
   // double q_stddev = 1.0;
   // double p_stddev = 1.0;

   // quantum_gaussian_source source(0.0, std_dev_VA, 0.0, std_dev_VA, q_stddev, p_stddev);

   // std::unique_ptr<source<gaussian_state, libbase::vector>> source;

   // Generate GM coherent states (via serialization)
   std::stringstream ss_src;
   ss_src << "# Mean of Q_Mean\n"
         << "0.0\n"
         << "# Stddev of Q_Mean\n"
         << "4.30116\n"
         << "# Mean of P_Mean\n"
         << "0.0\n"
         << "# Stddev of P_Mean\n"
         << "4.30116\n"
         << "# Stddev of Q\n"
         << "1.0\n"
         << "# Stddev of P\n"
         << "1.0\n";

   // Build source using the same pattern as gaussian_quantum_channel
   std::unique_ptr<libbase::serializable> s_ptr = libcomm::quantum_gaussian_source::create(ss_src);
   auto* src = dynamic_cast<libcomm::quantum_gaussian_source*>(s_ptr.get());
   BOOST_REQUIRE(src != nullptr);

   // Also view it through the source<> interface so we can seed/generate
   auto* source = dynamic_cast<libcomm::source<gaussian_state, libbase::vector>*>(src);

   // ----- Case 1: Used for non serialized
   // randgen r;
   // r.seed(7896);
   // source.seedfrom(r);

   // const int framesize = 1000; //1000; // Number of generated coherent states in a single frame.
   // vector<gaussian_state> source_sequence = source.generate_sequence(size_type<vector>(framesize));
   // std::cout << "Number of Generated Coherent States: " << framesize << std::endl;

   // Case 2: Used for serialized source
   randgen r;
   r.seed(7896);
   source->seedfrom(r);

   // const int framesize = 1000; //1000; // Number of generated coherent states in a single frame.
   // vector<gaussian_state> source_sequence = source->generate_sequence(size_type<vector>(framesize));
   // std::cout << "Number of Generated Coherent States: " << framesize << std::endl;

   // Case 3: Used a helper fn just to test out if I can use the source in the fn.
   // Pick a framesize once and reuse it everywhere
   const int framesize = 10;

   // Generate states via the helper (pass the interface by reference)
   vector<gaussian_state> source_sequence = generate_states(*source, framesize);

   // 2. Set Gaussian Quantum Channel and Identity Quantum Channel
   std::stringstream ss; // Serialized parameters of the quantum channel
   ss << "# Homodyne Detector Efficiency\n"
   << "0.606\n"    //"1.0\n"
   << "# Mean of the Gaussian Quantum Channel\n"
   << "0.0\n"
   << "# Transmittance T of the Gaussian Quantum Channel\n"
   << "0.302\n";

   // // Ideal case where they are both set to 1 - No noise - Ideal case
   // ss << "# Homodyne Detector Efficiency\n"
   // << "1.0\n"
   // << "# Mean of the Gaussian Quantum Channel\n"
   // << "0.0\n"
   // << "# Transmittance T of the Gaussian Quantum Channel\n"
   // << "1.0\n";

   // Setting Alice's identity quantum channel
   std::unique_ptr<quantum_channel> alice_channel = std::make_unique<identity_quantum_channel>();
   //     std::unique_ptr<quantum_channel> bob_channel = std::make_unique<gaussian_quantum_channel>();

   std::unique_ptr<libbase::serializable> ptr = libcomm::gaussian_quantum_channel::create(ss);
   auto* bob_channel = dynamic_cast<libcomm::gaussian_quantum_channel*>(ptr.get());
   BOOST_REQUIRE(bob_channel != nullptr);

   libbase::vector<double> params;
   params.init(1); // Only CLI parameter of the Quantum channel
   // double variance_VN = 1.0425099999999998; // Case 7
   //double variance_VN = 3.452019867549669; // using 1 _ Xtotal
   // double variance_VN =  1.04251; // with det eff of 1
   // double variance_VN = 1.04191506;  // Variance with det eff 0.606
   // 1.04191506; // Variance with det eff 0.606
   // double variance_VN = 0; // No noise case for now - Ideal case

   // double variance_VN = 1.0459999999999998
   // case of det eff = 1 and Transmittance = 1

   double SNR = 17.7558; // Linear not dB
   params(0) = SNR;

   double VA = src->get_VA(); // will be done in qkd_commsys object
   std::cout << "Value of VA from quantum gausssian source = " << VA << std::endl;

   bob_channel->set_parameters(params);
   auto all_params = bob_channel->get_parameters();
   bob_channel->set_VA(VA);

   // Print channel CLI parameter
   std::cout << all_params << std::endl;

   // 3. Set up QKD protocol
   randgen rng;
   rng.seed(17);
   std::unique_ptr<cvqkd_protocol> protocol = std::make_unique<cvqkd_protocol>();
   protocol->seedfrom(rng);

   // 4. Create observables for Bob
   std::vector<std::unique_ptr<observable<double>>> bob_observables = protocol->get_bob_observables(framesize);

   // Get Bob#s decision vector of his observables
   const libbase::vector<int>& decision_vector = protocol->get_decision_vector();

   std::cout<<"Printing Bob's generated observables"<< std::endl;
   for (int i = 0; i < framesize; ++i) {
      std::string type = (decision_vector(i) == 0) ? "Position" : "Momentum";
      std::cout << "Observable " << i << " (" << type << "): ";
   }

   // 5. Create observables for Alice
   std::vector<std::unique_ptr<observable<double>>> alice_observables = protocol->get_alice_observables(framesize, decision_vector); // Changed this method to accept two parameters: framesize and bob's decision vector

   // std::vector<std::unique_ptr<observable<double>>> alice_observables2 = protocol->get_alice_observables(framesize); // created this just to test the  observables for Alice. Eventually I will delete it.

   const libbase::vector<int>& alice_decision_vector = protocol->get_alice_decision_vector(); // This was just to be able to print the the type of observables that were generated.

   std::cout<<"\n Alice's generated Fake observables with noise"<< std::endl;
   for (int i = 0; i < framesize; ++i) {
      std::string type = (alice_decision_vector(i) == 0) ? "Position" : "Momentum";
      std::cout << "Observable " << i << " (" << type << "): ";
   }
   std::cout << std::endl;

   // 6. Initialising the measurement vectors for Alice and Bob
   libbase::vector<double> alice_measurements;
   libbase::vector<double> bob_measurements;
   alice_measurements.init(framesize);
   bob_measurements.init(framesize);

   // 7. Transmit Bob's observables through the channel and perform measurement
   for (int i = 0; i < framesize; ++i) {
      alice_observables[i]->transmit(*alice_channel);
      bob_observables[i]->transmit(*bob_channel);

      alice_measurements(i) = source_sequence(i).measure(*alice_observables[i]);
      bob_measurements(i) = source_sequence(i).measure(*bob_observables[i]);
   }

   // Print both measurement vectors.
   std::cout << "\nBob's measurements with noise: [";
   for (int i = 0; i < framesize; ++i) {
      std::cout << bob_measurements(i);
      if (i < framesize - 1) std::cout << ", ";
   }
   std::cout << "]" << std::endl;
   std::cout << std::endl;

   std::cout << "Alice's measurements with noise: [";
   for (int i = 0; i < framesize; ++i) {
      std::cout << alice_measurements(i);
      if (i < framesize - 1) std::cout << ", ";
   }
   std::cout << "]" << std::endl;
   std::cout << std::endl;
}

double qkd_fullcycletest(libcomm::quantum_gaussian_source& src) {
   double VA = src.get_VA();        // OK
   return VA;
}


   // // Set up the channels
   // std::unique_ptr<quantum_channel> alice_channel = std::make_unique<identity_quantum_channel>();
   // // std::unique_ptr<quantum_channel> bob_channel = std::make_unique<gaussian_quantum_channel>();

   // std::stringstream ss; // Serialized parameters of the quantum channel
   // ss << "# Homodyne Detector Efficiency\n"
   // << "0.606\n"    //"1.0\n"
   // << "# Mean of the Gaussian Quantum Channel\n"
   // << "0.0\n"
   // << "# Transmittance T of the Gaussian Quantum Channel\n"
   // << "0.302\n";

   // std::unique_ptr<libbase::serializable> ptr = libcomm::gaussian_quantum_channel::create(ss);
   // auto* bob_channel = dynamic_cast<libcomm::gaussian_quantum_channel*>(ptr.get());
   // BOOST_REQUIRE(bob_channel != nullptr);

   // libbase::vector<double> params;
   // params.init(1); // Only CLI parameter of the Quantum channel
   // double SNR = 17.7558; // Linear not dB
   // params(0) = SNR;
   // bob_channel->set_parameters(params);
   // auto all_params = bob_channel->get_parameters();
   // // Print channel (CLI parameter) SNR
   // std::cout << all_params << std::endl;

   // // Set up QKD protocol
   // randgen rng;
   // rng.seed(17);
   // std::unique_ptr<cvqkd_protocol> protocol = std::make_unique<cvqkd_protocol>();
   // protocol->seedfrom(rng);
   // std::cout << protocol->description() << std::endl;

   // // Setting qkd_commsys object
   // qkd_commsys< sys;
   // std::cout << sys.description() << std::endl;


BOOST_AUTO_TEST_CASE(test_qkd_commsys_object_up_until_measurement)
{
   std::cout << "\n*****Boost Test Case *****\n";
   std::cout << "\nTesting QKD Commsys up until Measurement\n";

   // 1) Build a config that matches qkd_commsys::serialize(std::istream&)
   std::stringstream cfg;
   cfg <<
      "# Version\n"
      "1\n"
      "# Frame size (# of quantum states in a frame)\n"
      "8\n"
      "## Alice's channel\n"
      "identity_quantum_channel\n"
      "## Bob's channel\n"
      "gaussian_quantum_channel\n"
      "# Homodyne Detector Efficiency\n"
      "0.606\n"
      "# Mean of the Gaussian Quantum Channel\n"
      "0.0\n"
      "# Transmittance T of the Gaussian Quantum Channel\n"
      "0.302\n"
      "## Postprocessing protocol\n"
      "cvqkd_protocol\n";

   // 2) Default-construct the system and load the config
   libcomm::qkd_commsys<libcomm::gaussian_state, double, libbase::vector> sys;
   sys.serialize(cfg);   // this populates alice_channel, bob_channel, protocol, framesize

   // 3) Set the CLI params (Bob’s SNR) through qkd_commsys
   const double SNR_linear = 17.7558; // linear (not dB)
   libbase::vector<double> cli;
   cli.init(sys.get_num_params());  // should be 1 when Alice is identity
   cli(0) = SNR_linear;             // index 0 -> Bob's SNR
   sys.set_parameters(cli);

   // 4) Print System Parameters of the QKD Commsys Object
   std::cout << "\n" << sys.description() << "\n\n";

   // 5) Verify CLI parameters of Quantum Channel of Bob
   auto back = sys.get_parameters();
   std::cout << "Print (SNR) CLI parameter of Bob's Quantum Channel " << back << std::endl;

   // 6) Create Gaussian Quantum Source
   std::stringstream ss_src;
   ss_src << "# Mean of Q_Mean\n"
         << "0.0\n"
         << "# Stddev of Q_Mean\n"
         << "4.30116\n"
         << "# Mean of P_Mean\n"
         << "0.0\n"
         << "# Stddev of P_Mean\n"
         << "4.30116\n"
         << "# Stddev of Q\n"
         << "1.0\n"
         << "# Stddev of P\n"
         << "1.0\n";

   // Build source using the same pattern as gaussian_quantum_channel
   std::unique_ptr<libbase::serializable> s_ptr = libcomm::quantum_gaussian_source::create(ss_src);
   auto* src = dynamic_cast<libcomm::quantum_gaussian_source*>(s_ptr.get());
   BOOST_REQUIRE(src != nullptr);

   randgen r;
   r.seed(7896);
   src->seedfrom(r);

   double VA = src->get_VA();
   std::cout << "\n Checking Modulation Variance of Source = " << VA << std::endl;

   // Set seed for qkd_commsys object
   randgen rng;
   rng.seed(17);
   sys.seedfrom(rng);

   // 5) Run the fullcycle that consumes a quantum_gaussian_source&
   auto [alice, bob] = sys.fullcycle(*src);

   // Print Measurement Vectors
   std::cout << "Alice measurements [size=" << alice.size() << "]: [";
   std::cout << std::fixed << std::setprecision(6);
   for (int i = 0; i < alice.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << alice(i);
   }
   std::cout << "]\n";

   std::cout << "Bob measurements [size=" << bob.size() << "]: [";
   for (int i = 0; i < bob.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << bob(i);
   }
   std::cout << "]\n";
}