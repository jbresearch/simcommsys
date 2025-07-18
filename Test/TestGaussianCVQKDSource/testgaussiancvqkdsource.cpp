/*!
 * \file
 * \brief Boost unit tests for quantum_gaussian_source and cvqkd_protocol
 *
 * Copyright (c) 2025 Aaron Abela
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>

#include "qkd/quantum_state.h"
#include "source/quantum_gaussian_source.h"
#include "qkd/observable/position_observable.h"
#include "qkd/observable/momentum_observable.h"
#include "qkd/qkd_protocol/cvqkd_protocol.h"
#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "random.h"
#include "truerand.h"

#include <iostream>
#include <memory>
#include <cmath>

using namespace libcomm;
using namespace libbase;

// Previous older tests
// BOOST_AUTO_TEST_CASE(test_single_gaussian_state) {
//     quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//     randgen r;
//     r.seed(0);
//     source.seedfrom(r);

//     gaussian_state state = source.generate_single();

//     std::cout << "\n[Generated a Single Gaussian State]" << std::endl;
//     std::cout << "q_mean = " << state.get_q() << ", p_mean = " << state.get_p() << std::endl;

//     BOOST_CHECK(std::isfinite(state.get_q()));
//     BOOST_CHECK(std::isfinite(state.get_p()));
// }

// BOOST_AUTO_TEST_CASE(test_gaussian_sequence) {
//     quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//     randgen r;
//     r.seed(7896);
//     source.seedfrom(r);

//     const int num_states = 10;
//     vector<gaussian_state> sequence = source.generate_sequence(size_type<vector>(num_states));

//     std::cout << "\n[Generated Sequence of Gaussian States]" << std::endl;
//     BOOST_CHECK_EQUAL(sequence.size(), num_states);

//     for (int i = 0; i < sequence.size(); ++i) {
//         std::cout << "State " << i << ": q = " << sequence(i).get_q()
//                   << ", p = " << sequence(i).get_p() << std::endl;

//         BOOST_CHECK(std::isfinite(sequence(i).get_q()));
//         BOOST_CHECK(std::isfinite(sequence(i).get_p()));
//     }
// }

// BOOST_AUTO_TEST_CASE(test_gaussian_measurement_with_position_observable) {
//     quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//     randgen r;
//     r.seed(42);
//     source.seedfrom(r);

//     gaussian_state state = source.generate_single();

//     double test_noise = 0.42;
//     position_observable obs;
//     obs.set_noise(test_noise);

//     double q_noisy = obs.measure(state); // measured value

//     std::cout << "\n[Position Observable Measurement]" << std::endl;
//     std::cout << "Measured value (q + noise) = " << q_noisy << std::endl;

//     BOOST_CHECK(std::isfinite(q_noisy));
// }

// BOOST_AUTO_TEST_CASE(test_gaussian_measurement_with_momentum_observable) {
//     quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//     randgen r;
//     r.seed(84);
//     source.seedfrom(r);

//     gaussian_state state = source.generate_single();

//     double test_noise = 0.57;
//     momentum_observable obs;
//     obs.set_noise(test_noise);

//     double p_noisy = obs.measure(state); // measured value

//     std::cout << "\n[Momentum Observable Measurement]" << std::endl;
//     std::cout << "Measured value (p + noise) = " << p_noisy << std::endl;

//     BOOST_CHECK(std::isfinite(p_noisy));
// }

// BOOST_AUTO_TEST_CASE(test_cvqkd_protocol_bob_observables_decision_vector) {
//     cvqkd_protocol cvqkdprotocol;
//     randgen rng;
//     rng.seed(12345);
//     cvqkdprotocol.seedfrom(rng);

//     const int framesize = 10;
//     std::vector<std::unique_ptr<observable<double>>> bob_observables = cvqkdprotocol.get_bob_observables(framesize);
//     const libbase::vector<int>& decision_vector = cvqkdprotocol.get_decision_vector();

//     BOOST_CHECK_EQUAL(bob_observables.size(), framesize);
//     BOOST_CHECK_EQUAL(decision_vector.size(), framesize);

//     std::cout << "\n[CVQKD Bob Observables and Decision Vector]" << std::endl;
//     for (int i = 0; i < framesize; ++i) {
//         std::string type = (decision_vector(i) == 0) ? "Position" :
//                            (decision_vector(i) == 1) ? "Momentum" : "Unknown";

//         std::cout << "Observable " << i << ": " << type << std::endl;

//         if (decision_vector(i) == 0) {
//             BOOST_CHECK(dynamic_cast<position_observable*>(bob_observables[i].get()) != nullptr);
//         } else if (decision_vector(i) == 1) {
//             BOOST_CHECK(dynamic_cast<momentum_observable*>(bob_observables[i].get()) != nullptr);
//         } else {
//             BOOST_FAIL("Invalid value in decision vector.");
//         }
//     }
// }

// BOOST_AUTO_TEST_CASE(test_cvqkd_protocol_bob_observables_decision_vector_using_pointer) {
//     std::unique_ptr<cvqkd_protocol> protocol = std::make_unique<cvqkd_protocol>();
//     std::cout<< "\nThis test uses pointer and pointer referencing " << std::endl;
//     randgen rng;
//     rng.seed(1237);
//     protocol->seedfrom(rng);

//     const int framesize = 10;
//     std::vector<std::unique_ptr<observable<double>>> bob_observables = protocol->get_bob_observables(framesize);
//     const libbase::vector<int>& decision_vector = protocol->get_decision_vector();

//     BOOST_CHECK_EQUAL(bob_observables.size(), framesize);
//     BOOST_CHECK_EQUAL(decision_vector.size(), framesize);

//     std::cout << "\n[CVQKD Bob Observables and Decision Vector]" << std::endl;
//     for (int i = 0; i < framesize; ++i) {
//         std::string type = (decision_vector(i) == 0) ? "Position" :
//                            (decision_vector(i) == 1) ? "Momentum" : "Unknown";

//         std::cout << "Observable " << i << ": " << type << std::endl;

//         if (decision_vector(i) == 0) {
//             BOOST_CHECK(dynamic_cast<position_observable*>(bob_observables[i].get()) != nullptr);
//         } else if (decision_vector(i) == 1) {
//             BOOST_CHECK(dynamic_cast<momentum_observable*>(bob_observables[i].get()) != nullptr);
//         } else {
//             BOOST_FAIL("Invalid value in decision vector.");
//         }
//     }
// }

// BOOST_AUTO_TEST_CASE(test_gaussian_channel_on_observables) {
//     std::cout << "\n[Gaussian Quantum Channel Applied to Observables]" << std::endl;

//     // 1. Create Gaussian quantum channel
//     std::unique_ptr<quantum_channel> channel = std::make_unique<gaussian_quantum_channel>();
//     // gaussian_quantum_channel channel;
//     randgen rng;
//     rng.seed(12);
//     channel->seedfrom(rng);

//     std::cout << "Before printing the description of the channel. Testing with pointer referencing notation. " << std::endl;

//     std::cout << "Channel description: " << channel->description() << std::endl;

//     std::cout << "After printing the description of the channel " << std::endl;

//     // Set noise mean and stddev (mean is serialized, stddev is CLI param)
//     std::istringstream sin("Mean of the Gaussian Quantum Channel\n0.0\n");
//     channel->serialize(sin);  // Deserialize from stream

//     vector<double> params;
//     params.init(1);
//     params(0) = 0.8;  // standard deviation of the noise, CLI parameter

//     channel->set_parameters(params);

//     std::cout<<"getting the set parameters: " << channel->get_parameters() << std::endl;

//     // 2. Generate observablesW
//     const int framesize = 2;
//     cvqkd_protocol protocol;
//     protocol.seedfrom(rng);
//     std::vector<std::unique_ptr<observable<double>>> observables = protocol.get_bob_observables(framesize);
//     const vector<int>& decision_vector = protocol.get_decision_vector();

//     // 3. Transmit observables through channel and print noise
//     for (int i = 0; i < framesize; ++i) {
//         observable<double>* obs_ptr = observables[i].get();
//         std::string type = (decision_vector(i) == 0) ? "Position" : "Momentum";

//         // Print type before transmission
//         std::cout << "Observable " << i << " (" << type << "): ";

//         // Transmit using the channel
//         obs_ptr->transmit(*channel);

//         // Retrieve and print the noise added
//         double noise_val = 0.0;
//         if (auto* pos_obs = dynamic_cast<position_observable*>(obs_ptr)) {
//             noise_val = pos_obs->get_noise();
//         } else if (auto* mom_obs = dynamic_cast<momentum_observable*>(obs_ptr)) {
//             noise_val = mom_obs->get_noise();
//         }

//         std::cout << "Noise added = " << noise_val << std::endl;

//         // BOOST_CHECK(std::isfinite(noise_val));
//     }
// }

// BOOST_AUTO_TEST_CASE(bobs_measurement_test) {
//     std::cout << "\n[Gaussian Quantum Channel Applied to Observables]" << std::endl;

//     // 0. Create a source of GM coherent states
//      quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//     randgen r;
//     r.seed(7896);
//     source.seedfrom(r);

//     const int framesize = 2; // Number of generated coherent states in a single frame

//     vector<gaussian_state> source_sequence = source.generate_sequence(size_type<vector>(framesize));
//     std::cout << "Number of Generated GM Coherent States: " << framesize << std::endl;

//     std::cout << "\n[Testing Bob's Measurement Vector]" << std::endl;
//     BOOST_CHECK_EQUAL(source_sequence.size(), framesize);

//     // for (int i = 0; i < source_sequence.size(); ++i) {
//     //     std::cout << "State " << i << ": q = " << source_sequence(i).get_q()
//     //               << ", p = " << source_sequence(i).get_p() << std::endl;
//     // }

//     // 1. Create Gaussian quantum channel
//     std::unique_ptr<quantum_channel> channel = std::make_unique<gaussian_quantum_channel>();
//     randgen rng;
//     rng.seed(12);
//     channel->seedfrom(rng);

//     std::cout << "Channel description: " << channel->description() << std::endl;

//     // Set noise mean and stddev (mean is serialized, stddev is CLI param)
//     // std::istringstream sin("Mean of the Gaussian Quantum Channel\n0.0\nTransmittance of the Gaussian Quantum Channel\n0.63\nHomodyne Detector Efficiency\n0.6\n");
//     // channel->serialize(sin);  // Deserialize from stream

//     // Set all 4 parameters as CLI parameters for now
//     libbase::vector<double> params;
//     params.init(4);
//     params(0) = 0.0; // Mean of the noise
//     params(1) = 0.8;  // standard deviation of the noise, CLI parameter
//     params(2) = 0.63; // Transmittance T
//     params(3) = 0.6; // Homodyne Detector efficiency

//     channel->set_parameters(params);

//     std::cout<<"Getting the set parameters: " << channel->get_parameters() << std::endl;

//     // 2. Set QKD protocol and Generate observables
//     cvqkd_protocol protocol;
//     protocol.seedfrom(rng);

//     // std::unique_ptr<quantum_channel> channel = std::make_unique<gaussian_quantum_channel>();

//     std::vector<std::unique_ptr<observable<double>>> observables = protocol.get_bob_observables(framesize);
//     const vector<int>& decision_vector = protocol.get_decision_vector();

//     // 3. Create allocate vector for measurements on Bob's end.
//     libbase::vector<double> bob_measurements;
//     bob_measurements.init(framesize);

//     // 4. Transmit observables through channel and print noise
//     for (int i = 0; i < framesize; ++i) {
//         observable<double>* obs_ptr = observables[i].get();
//         std::string type = (decision_vector(i) == 0) ? "Position" : "Momentum";

//         // Print type before transmission
//         std::cout << "Observable " << i << " (" << type << "): ";

//         // Transmit using the channel
//         obs_ptr->transmit(*channel);

//         // Retrieve and print the noise added
//         double noise_val = 0.0;
//         double transmittance_val = 0.0;
//         double detector_eff_val = 0.0;
//         if (auto* pos_obs = dynamic_cast<position_observable*>(obs_ptr)) {
//             noise_val = pos_obs->get_noise();
//             transmittance_val = pos_obs->get_transmittance();
//             detector_eff_val = pos_obs->get_detector_eff();
//         } else if (auto* mom_obs = dynamic_cast<momentum_observable*>(obs_ptr)) {
//             noise_val = mom_obs->get_noise();
//             transmittance_val = mom_obs->get_transmittance();
//             detector_eff_val = mom_obs->get_detector_eff();
//         }

//         std::cout << "Noise added = " << noise_val << std::endl;
//         std::cout << "Transmittance = " << transmittance_val << std::endl;
//         std::cout << "Homodyne Detector Efficiency = " << detector_eff_val << std::endl;

//         bob_measurements(i) = source_sequence(i).measure(*observables[i]);

//         std::cout << "Bobs measured value = " <<  bob_measurements(i) << std::endl;
//     }
// }

// BOOST_AUTO_TEST_CASE(bobs_measurement_test_fullcycle_style) {
//     std::cout << "\n[Testing Bob's Measurement Vector — Fullcycle Style]" << std::endl;

//     // 1. Generate GM coherent states
//     quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//     randgen r;
//     r.seed(7896);
//     source.seedfrom(r);

//     const int framesize = 4;
//     vector<gaussian_state> source_sequence = source.generate_sequence(size_type<vector>(framesize));
//     BOOST_CHECK_EQUAL(source_sequence.size(), framesize);
//     std::cout << "Number of Generated GM Coherent States: " << framesize << std::endl;

//     // 2. Create Bob's Gaussian quantum channel
//     // (Alice side commented for now)
//     std::unique_ptr<quantum_channel> alice_channel = std::make_unique<identity_quantum_channel>();
//     std::unique_ptr<quantum_channel> bob_channel = std::make_unique<gaussian_quantum_channel>();

//     randgen rng;
//     // rng.seed(12);
//     rng.seed(17);
//     // alice_channel->seedfrom(rng);
//     bob_channel->seedfrom(rng);

//     // Set channel parameters (mean, stddev, transmittance, detector eff)

//     // Deserialize serialized parameter (noise_mean)
//     std::istringstream sin(R"(
//     Homodyne Detector Efficiency
//     0.6
//     )");
//     bob_channel->serialize(sin);

//     libbase::vector<double> channel_params;
//     channel_params.init(3); // previously was 4
//     channel_params(0) = 0.0;   // noise mean
//     channel_params(1) = 0.8;   // stddev
//     channel_params(2) = 0.63;  // transmittance
//     // channel_params(3) = 0.6;   // detector efficiency

//     // alice_channel->set_parameters(channel_params);
//     bob_channel->set_parameters(channel_params);
//     std::cout << "Channel parameters set for Bob." << std::endl;

//     // 3. Set up QKD protocol
//     std::unique_ptr<cvqkd_protocol> protocol = std::make_unique<cvqkd_protocol>();
//     protocol->seedfrom(rng);

//     std::vector<std::unique_ptr<observable<double>>> bob_observables = protocol->get_bob_observables(framesize);
//     const libbase::vector<int>& decision_vector = protocol->get_decision_vector();


//     // 4. Print Bob's Observables
//     std::cout<<"Dummy Test: Printing Bob's generated observables in test cpp file"<< std::endl;
//     for (int i = 0; i < framesize; ++i) {
//         std::string type = (decision_vector(i) == 0) ? "Position" : "Momentum";
//         std::cout << "Observable " << i << " (" << type << "): ";
//     }

//     std::vector<std::unique_ptr<observable<double>>> alice_observables = protocol->get_alice_observables(framesize, decision_vector); // Changed this method to accept two parameters: framesize and bob's decision vector

//     // To delete this. Just for dummy purposes.
//     // const libbase::vector<int>& alice_decision_vector = protocol->get_alice_decision_vector();

//     libbase::vector<double> alice_measurements;
//     libbase::vector<double> bob_measurements;
//     alice_measurements.init(framesize);
//     bob_measurements.init(framesize);

//     // 5. Transmit Bob's observables through the channel and perform measurement
//     for (int i = 0; i < framesize; ++i) {
//         // std::string type = (decision_vector(i) == 0) ? "Position" : "Momentum";
//         // std::cout << "Observable " << i << " (" << type << "): " << std::endl;

//         alice_observables[i]->transmit(*alice_channel);
//         bob_observables[i]->transmit(*bob_channel);

//         alice_measurements(i) = source_sequence(i).measure(*alice_observables[i]);

//         bob_measurements(i) = source_sequence(i).measure(*bob_observables[i]);

//         // Debug print for noise
//         double noise_val = 0.0, T_val = 0.0, eta_val = 0.0;
//         if (auto* obs = dynamic_cast<position_observable*>(bob_observables[i].get())) {
//             noise_val = obs->get_noise();
//             T_val     = obs->get_transmittance();
//             eta_val   = obs->get_detector_eff();
//         } else if (auto* obs = dynamic_cast<momentum_observable*>(bob_observables[i].get())) {
//             noise_val = obs->get_noise();
//             T_val     = obs->get_transmittance();
//             eta_val   = obs->get_detector_eff();
//         }

//         std::cout << "Noise = " << noise_val << ", T = " << T_val << ", η = " << eta_val << std::endl;
//         std::cout << "Bob's measured value = " << bob_measurements(i) << std::endl;

//         std::cout << "Alice's measured value = " << alice_measurements(i) << std::endl;
//     }
// }

// // Place this above your BOOST_AUTO_TEST_CASE
// static std::unique_ptr<libbase::serializable> create_gaussian_quantum_channel(std::istream& sin) {
//     auto obj = std::make_unique<libcomm::gaussian_quantum_channel>();
//     obj->serialize(sin);
//     return obj;
// }

BOOST_AUTO_TEST_CASE(test_gaussian_quantum_channel_serialisation)
{
   std::stringstream ss;
   ss << "Homodyne Detector Efficiency\n"
      << "0.6\n";

   std::unique_ptr<libbase::serializable> ptr = libcomm::gaussian_quantum_channel::create(ss);
   auto* channel = dynamic_cast<libcomm::gaussian_quantum_channel*>(ptr.get());
   BOOST_REQUIRE(channel != nullptr);

   libbase::vector<double> params;
   params.init(3);
   params(0) = 0.0;
   params(1) = 0.8;
   params(2) = 0.63;

   channel->set_parameters(params);
   auto all_params = channel->get_parameters();

   std::cout << all_params << std::endl;
   //  BOOST_CHECK_CLOSE(all_params(3), 0.6, 1e-6); // HDE must match serialized value


   // Set up QKD protocol
   int framesize = 2;
   randgen rng;
   // rng.seed(12);
   rng.seed(17);
   std::unique_ptr<cvqkd_protocol> protocol = std::make_unique<cvqkd_protocol>();
   protocol->seedfrom(rng);

   std::vector<std::unique_ptr<observable<double>>> bob_observables = protocol->get_bob_observables(framesize);
   const libbase::vector<int>& decision_vector = protocol->get_decision_vector();

   // Transmit for Bob's observables.
   for (int i = 0; i < framesize; ++i) {
        std::string type = (decision_vector(i) == 0) ? "Position" : "Momentum";
        std::cout << "Observable " << i << " (" << type << "): " << std::endl;

        bob_observables[i]->transmit(*channel);
   }

}


