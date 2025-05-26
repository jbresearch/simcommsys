/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#define BOOST_TEST_NO_MAIN
#include <boost/test/included/unit_test.hpp>

#include "qkd/quantum_state.h"
#include "source/quantum_gaussian.h"
#include "source.h"
#include "serializer.h"

#include "qkd/quantum_channel.h"
#include "qkd/position_observable.h"
#include "qkd/momentum_observable.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>

using namespace libcomm;
using namespace libbase;
using namespace std;

int run_boost_tests(int argc, char* argv[]);  // Forward declaration

int main(int argc, char* argv[]) {

    // // -------- Test 1: Generating a single coherent state --------
    // const double mean_p_mean = 0.0;
    // const double stddev_p_mean = std::sqrt(10); // 0.3162; // square root of V_A = 0.1
    // const double mean_q_mean = 0.0;
    // const double stddev_q_mean = std::sqrt(10); //0.3162; // square root of V_A = 0.1

    // quantum_gaussian single_source(mean_q_mean, stddev_q_mean, mean_p_mean, stddev_p_mean);
    // gaussian_state state = single_source.generate_single();

    // double q_mean = state.get_q_mean();
    // double p_mean = state.get_p_mean();

    // cout << "Test 1 - Generated a Single Gaussian state:\n";
    // std::cout << "Generated state parameters:\n";
    // std::cout << " q_mean = " << q_mean << "\n";
    // std::cout << " p_mean = " << p_mean << "\n";


    // // Test 2 - Generate a sequence of coherent states with their own respective q_mean and p_mean
    // int num_states = 5;
    // libbase::size_type<libbase::vector> blocksize(num_states);

    // const double mean_p_mean2 = 0.0;
    // const double stddev_p_mean2 = 10.0;
    // const double mean_q_mean2 = 0.0;
    // const double stddev_q_mean2 = 10.0;

    // auto src = std::make_unique<quantum_gaussian>(mean_q_mean2, stddev_q_mean2, mean_p_mean2, stddev_p_mean2);
    // libbase::vector<gaussian_state> gaussian_seq = src->generate_sequence(blocksize);

    // cout << "\nTest 2 - Generate Sequence of Gaussian Coherent States\n";
    // for (int i = 0; i < gaussian_seq.size(); ++i) {
    //     gaussian_state& s = gaussian_seq(i);
    //     cout << "State " << i << ": q_mean = " << s.get_q_mean() << ", p_mean = " << s.get_p_mean() << "\n";
    // }

    // // // -------- Extra Test  --------
    // // cout << "\nTest 4 - Choosing between p and q for each generated state (50% chance)\n";
    // // libbase::vector<double> selected_quadrature(gaussian_seq.size());
    // // std::mt19937 rng(42);
    // // std::uniform_real_distribution<> dist(0.0, 1.0);

    // // for (int i = 0; i < gaussian_seq.size(); ++i) {
    // //     gaussian_state& s = gaussian_seq(i);
    // //     selected_quadrature(i) = (dist(rng) < 0.5) ? s.get_q() : s.get_p();
    // // }

    // // cout << "Selected values (q or p):\n";
    // // for (int i = 0; i < selected_quadrature.size(); ++i) {
    // //     cout << i << ": " << selected_quadrature(i) << "\n";
    // // }

    // // -------- Test 4 - Transmit the GM Coherent State through a Gaussian Quantum Channel --------
    // cout << "\nTest 4 - Transmit the GM Coherent State through a Gaussian Quantum Channel without using text file/serializer \n";
    // // Step 2: Wrap them in observables
    // position_observable q_obs(q_mean);
    // momentum_observable p_obs(p_mean);

    // std::cout << "[Before Channel] q = " << q_obs.value << ", p = " << p_obs.value << std::endl;

    // // Step 3: Create and configure the Gaussian quantum channel
    // gaussian_quantum_channel channel;

    // libbase::vector<double> params;
    // params.init(6);
    // params(0) = 1.0;   // N_0
    // params(1) = 0.2;   // alpha
    // params(2) = 0.6;   // detector efficiency
    // params(3) = 70.0;  // distance in km
    // params(4) = 0.01;  // excess noise
    // params(5) = 0.1;   // electronic noise
    // channel.set_parameters(params);

    // // Step 4: Apply noise
    // channel.transmit(q_obs);
    // channel.transmit(p_obs);
    // channel.get_parameters();

    // channel.print_parameters();
    // // std::cout << "Transmittance: " << channel.get_parameters()(6) << "\n";
    // // std::cout << "V_N: " << channel.get_parameters()(7) << "\n";

    // std::cout << "[After Channel]  q_mean = " << q_obs.value << ", p_mean = " << p_obs.value << std::endl;

    // // -------- Test 5 - Boost Test Case --------
    cout << "\nTest 5 - Boost Test Case: \n";
    cout << "\nRunning Boost test case...\n";
    return run_boost_tests(argc, argv);


}

// ------------------------------------
// BOOST TEST CASE
// ------------------------------------
BOOST_AUTO_TEST_CASE(test_gaussian_source_serialisation)
{
    //  -------- 1. Set up a quantum_gaussian source with fixed seed
    std::stringstream ss;
    ss << "# Mean of Q_Mean\n"
       << "0\n"
       << "# Stddev of Q_Mean\n"
       << "4.2\n"
       << "# Mean of P_Mean\n"
       << "0\n"
       << "# Stddev of P_Mean\n"
       << "4.2\n";

    std::unique_ptr<serializable> ptr = quantum_gaussian::create(ss);
    auto* source = dynamic_cast<quantum_gaussian*>(ptr.get());
    BOOST_REQUIRE_MESSAGE(source != nullptr, "Failed to deserialize quantum_gaussian");

    // source->set_seed(123);  // Fixed seed

    //  Test case for a single state
    // gaussian_state state = source->generate_single();
    // double q_mean = state.get_q_mean();
    // double p_mean = state.get_p_mean();
    // std::cout << "\n[BOOST TEST] Generated Gaussian State:\n"
    //           << "q_mean = " << q_mean << ", p_mean = " << p_mean << std::endl;


    //  -------- 2. Generate a sequence of N Gaussian states
    std::cout << "\n[BOOST TEST] Alice's Gaussian Source Parameters:\n";
    std::cout << source->description();
    std::cout << endl << endl;

    const int num_states = 10;
    libbase::size_type<libbase::vector> blocksize(num_states);
    libbase::vector<gaussian_state> state_seq = source->generate_sequence(blocksize);


    // std::cout << "\n[BOOST TEST] Transmit State through the Gaussian Channel:" << std::endl;

    //  -------- 3. Create the Gaussian quantum channel with fixed parameters
    std::stringstream ss_G_QC;
    ss_G_QC << "# Vaccuum/Shot Noise (N_0)\n"
        << "1.0\n"
        << "# Attenuation coefficient (Alpha)\n"
        << "0.2\n"
        << "# Detector Efficiency (eta)\n"
        << "0.6\n"
        << "# Distance in Km (l)\n"
        << "70.0\n"
        << "# Excess Noise (xi)\n"
        << "0.01\n"
        << "# Electric Noise (v_el)\n"
        << "0.1\n";

    // // Test case for a single state - Wrap q_mean and p_mean them in observables
    // position_observable q_obs(q_mean);
    // momentum_observable p_obs(p_mean);

    //  Deserialize ss_G_QC to get the quantum_gaussian_channel
    std::unique_ptr<serializable> ptr2 = gaussian_quantum_channel::create(ss_G_QC);
    auto* quantum_channel = dynamic_cast<gaussian_quantum_channel*>(ptr2.get());

    quantum_channel->print_parameters();

    // Test case for a single state
    // std::cout << "[Before Channel] q = " << q_obs.value << ", p = " << p_obs.value << std::endl;

    // // Create and configure the Gaussian quantum channel
    // // libbase::vector<double> params;
    // // params.init(6);
    // // channel.set_parameters(params);

    // // Apply noise
    // channel->transmit(q_obs);
    // channel->transmit(p_obs);

    // channel->print_parameters();
    // // std::cout << "Transmittance: " << channel.get_parameters()(6) << "\n";
    // // std::cout << "V_N: " << channel.get_parameters()(7) << "\n";

    // std::cout << "[After Channel]  q_mean = " << q_obs.value << ", p_mean = " << p_obs.value << std::endl;


    // -------- 4. Transmit each state through the gaussian channel
    std::cout << "\n[BOOST TEST] Transmitting sequence through Gaussian Quantum Channel:\n";
    std::cout << std::left
            << std::setw(6) << "Index"
            << std::setw(16) << "q_mean_in"
            << std::setw(16) << "p_mean_in"
            << std::setw(16) << "q_mean_out"
            << std::setw(16) << "p_mean_out" << "\n";

    std::cout << std::string(70, '-') << "\n";

    for (int i = 0; i < state_seq.size(); ++i) {
        gaussian_state& state = state_seq(i);

        double q_in = state.get_q_mean();
        double p_in = state.get_p_mean();

        position_observable q_obs(q_in);
        momentum_observable p_obs(p_in);

        quantum_channel->transmit(q_obs);
        quantum_channel->transmit(p_obs);

        double q_out = q_obs.value;
        double p_out = p_obs.value;

        std::cout << std::left
                << std::setw(6) << i
                << std::setw(16) << q_in
                << std::setw(16) << p_in
                << std::setw(16) << q_out
                << std::setw(16) << p_out << "\n";
    }
}

// ------------------------------------
// Required by BOOST_TEST_NO_MAIN
// ------------------------------------
boost::unit_test::test_suite* init_unit_test(int, char*[]) {
    return nullptr;  // Use default auto-registered test suite
}

int run_boost_tests(int argc, char* argv[]) {
    return boost::unit_test::unit_test_main(&init_unit_test, argc, argv);
}