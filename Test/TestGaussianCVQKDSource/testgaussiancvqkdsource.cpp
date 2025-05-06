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

#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <random>

using namespace libcomm;
using namespace libbase;
using namespace std;

int run_boost_tests(int argc, char* argv[]);  // Forward declaration

int main(int argc, char* argv[]) {

    // -------- Test 1: Generating a single coherent state --------
    const double mean_p_mean = 0.0;
    const double stddev_p_mean = 1.0;
    const double mean_q_mean = 0.0;
    const double stddev_q_mean = 1.0;

    quantum_gaussian single_source(mean_q_mean, stddev_q_mean, mean_p_mean, stddev_p_mean);
    gaussian_state state = single_source.generate_single();

    double q_mean = state.get_q_mean();
    double p_mean = state.get_p_mean();

    cout << "Test 1 - Generated a Single Gaussian state:\n";
    std::cout << "Generated state parameters:\n";
    std::cout << " q_mean = " << q_mean << "\n";
    std::cout << " p_mean = " << p_mean << "\n";

    // Test 2 - Add 'noise' to the q_mean and p_mean (dummy test)
    cout << "Test 2 - Adding Noise to the Quadrature Components p and q:\n";
    double noisy_q_mean = q_mean + 0.5;
    double noisy_p_mean = p_mean + 0.5;
    cout << "Noisy q_mean = " << noisy_q_mean << "\n";
    cout << "Noisy p_mean = " << noisy_p_mean << "\n";

    // Test 3 - Generate a sequence of coherent states with their own respective q_mean and p_mean
    int num_states = 5;
    libbase::size_type<libbase::vector> blocksize(num_states);

    const double mean_p_mean2 = 0.0;
    const double stddev_p_mean2 = 10.0;
    const double mean_q_mean2 = 0.0;
    const double stddev_q_mean2 = 10.0;

    auto src = std::make_unique<quantum_gaussian>(mean_q_mean2, stddev_q_mean2, mean_p_mean2, stddev_p_mean2);
    libbase::vector<gaussian_state> gaussian_seq = src->generate_sequence(blocksize);

    cout << "\nTest 3 - Generate Sequence of Gaussian Coherent States\n";
    for (int i = 0; i < gaussian_seq.size(); ++i) {
        gaussian_state& s = gaussian_seq(i);
        cout << "State " << i << ": q_mean = " << s.get_q_mean() << ", p = " << s.get_p_mean() << "\n";
    }

    // // -------- Extra Test  --------
    // cout << "\nTest 4 - Choosing between p and q for each generated state (50% chance)\n";
    // libbase::vector<double> selected_quadrature(gaussian_seq.size());
    // std::mt19937 rng(42);
    // std::uniform_real_distribution<> dist(0.0, 1.0);

    // for (int i = 0; i < gaussian_seq.size(); ++i) {
    //     gaussian_state& s = gaussian_seq(i);
    //     selected_quadrature(i) = (dist(rng) < 0.5) ? s.get_q() : s.get_p();
    // }

    // cout << "Selected values (q or p):\n";
    // for (int i = 0; i < selected_quadrature.size(); ++i) {
    //     cout << i << ": " << selected_quadrature(i) << "\n";
    // }

    // -------- Test 4 - Boost Test Case --------
    cout << "\nTest 4 - Boost Test Case: \n";
    cout << "\nRunning Boost test case...\n";
    return run_boost_tests(argc, argv);
}

// ------------------------------------
// BOOST TEST CASE
// ------------------------------------
BOOST_AUTO_TEST_CASE(test_gaussian_source_serialisation)
{
    std::stringstream ss;
    ss << "quantum_gaussian\n"
       << "# Mean of Q_Mean\n"
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

    source->set_seed(123);  // Fixed seed
    gaussian_state state = source->generate_single();

    std::cout << "\n[BOOST TEST] Generated Gaussian State:\n"
              << "q_mean = " << state.get_q_mean() << ", p_mean = " << state.get_p_mean() << "\n";

    BOOST_TEST(std::abs(state.get_q()) < 20.0);
    BOOST_TEST(std::abs(state.get_p()) < 20.0);
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