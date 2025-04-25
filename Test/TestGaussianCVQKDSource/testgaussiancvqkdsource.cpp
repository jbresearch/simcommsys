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
#include "source/gaussian.h"
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

    // -------- Test 1 --------
    const double mean = 0.0;
    const double stddev = 1.0;
    const unsigned int seed = 42;

    gaussian<gaussian_state> state_source(mean, stddev);
    state_source.set_seed(seed);
    gaussian_state state = state_source.generate_single();

    double p = state.get_p();
    double q = state.get_q();

    cout << "Test 1 - Generated a Single Gaussian state:\n";
    cout << "Mean = " << mean << "\n";
    cout << "Stddev = " << stddev << "\n";
    cout << "q = " << q << "\n";
    cout << "p = " << p << "\n";

    // -------- Test 2 --------
    cout << "\nTest 2 - Adding Noise to the Quadrature Components p and q:\n";
    double noisy_p = p + 0.5;
    double noisy_q = q + 0.5;
    cout << "Noisy q = " << noisy_q << "\n";
    cout << "Noisy p = " << noisy_p << "\n";

    // -------- Test 3 --------
    int num_states = 5;
    libbase::size_type<libbase::vector> blocksize(num_states);
    const double mean_seq = 0.0;
    const double stddev_seq = 20.0;

    auto src = std::make_unique<gaussian<gaussian_state>>(mean_seq, stddev_seq);
    libbase::vector<gaussian_state> gaussian_seq = src->generate_sequence(blocksize);

    cout << "\nTest 3 - Generate Sequence of Gaussian Coherent States\n";
    cout << "Mean of Sequence: " << mean_seq << "\n";
    cout << "StdDev of Sequence: " << stddev_seq << "\n";

    for (int i = 0; i < gaussian_seq.size(); ++i) {
        gaussian_state& s = gaussian_seq(i);
        cout << "State " << i << ": q = " << s.get_q() << ", p = " << s.get_p() << "\n";
    }

    // -------- Test 4 --------
    cout << "\nTest 4 - Choosing between p and q for each generated state (50% chance)\n";
    libbase::vector<double> selected_quadrature(gaussian_seq.size());
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < gaussian_seq.size(); ++i) {
        gaussian_state& s = gaussian_seq(i);
        selected_quadrature(i) = (dist(rng) < 0.5) ? s.get_q() : s.get_p();
    }

    cout << "Selected values (q or p):\n";
    for (int i = 0; i < selected_quadrature.size(); ++i) {
        cout << i << ": " << selected_quadrature(i) << "\n";
    }

    // -------- Test 5 - Boost Test Case --------
    cout << "\nTest 5 - Boost Test Case: \n";
    cout << "\nRunning Boost test case...\n";
    return run_boost_tests(argc, argv);
}

// ------------------------------------
// BOOST TEST CASE
// ------------------------------------
BOOST_AUTO_TEST_CASE(test_gaussian_source_serialisation)
{
    std::stringstream ss;
    ss << "gaussian<gaussian_state,vector>\n"
       << "# Mean\n"
       << "0\n"
       << "# Variance\n"
       << "4.2\n";

    std::unique_ptr<serializable> ptr = gaussian<gaussian_state>::create(ss);
    auto* source = dynamic_cast<gaussian<gaussian_state>*>(ptr.get());
    BOOST_REQUIRE_MESSAGE(source != nullptr, "Failed to deserialize gaussian<gaussian_state>");

    source->set_seed(123);  // Fixed seed for reproducibility
    gaussian_state state = source->generate_single();

    std::cout << "\n[BOOST TEST] Generated Gaussian State:\n"
              << "q = " << state.get_q() << ", p = " << state.get_p() << "\n";

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