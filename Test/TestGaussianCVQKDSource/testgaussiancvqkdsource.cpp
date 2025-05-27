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
#include "truerand.h"

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

    // // -------- Boost Test Case --------
    cout << "\nBoost Test Case: \n";
    cout << "\nRunning Boost test case...\n";
    return run_boost_tests(argc, argv);
}

// ------------------------------------
// BOOST TEST CASE
// ------------------------------------
BOOST_AUTO_TEST_CASE(test_gaussian_source_serialisation)
{
    // Setup quantum_gaussian source
    std::stringstream ss;
    ss << "# Mean of Q_Mean\n"
    << "0\n"
    << "# Stddev of Q_Mean\n"
    << "10\n"
    << "# Mean of P_Mean\n"
    << "0\n"
    << "# Stddev of P_Mean\n"
    << "10\n"
    << "# Stddev of P\n"
    << "1\n"
    << "# Stddev of Q\n"
    << "1\n";

    std::unique_ptr<serializable> ptr = quantum_gaussian::create(ss);
    auto* gaussian_source = dynamic_cast<quantum_gaussian*>(ptr.get());
    BOOST_REQUIRE_MESSAGE(gaussian_source != nullptr, "Failed to deserialize quantum_gaussian");

    // Seed setup
    libbase::truerand trng; // Idea taken from constructor of montecarlo.h
    libbase::int32u seed = trng.ival();
    // libbase::int32u seed = 2871727006;
    libbase::randgen prng; // Idea taken from seed_experiment() from Montecarlo.cpp
    prng.seed(seed);
    std::cerr << "[TEST] Random seed used for PRNG: " << seed << std::endl;

    // Seed source and generate state
    gaussian_source->seedfrom(prng);
    std::cout << "[TEST] Source Description: " << gaussian_source->description() << std::endl;

    // gaussian_state coherent_state = gaussian_source->generate_single();
    // coherent_state.seedfrom(prng);

    const int num_states = 10;
    libbase::size_type<libbase::vector> blocksize(num_states);
    libbase::vector<gaussian_state> coherent_state_seq = gaussian_source->generate_sequence(blocksize);

    for (int i = 0; i < coherent_state_seq.size(); ++i) {
        // gaussian_state& state = coherent_state_seq(i);

        // Performing measurement without passing it through the quantum channel
        double q_val = coherent_state_seq(i).get_q();
        double p_val = coherent_state_seq(i).get_p();
        std::cout<< "State Number:" << i << std::endl;
        std::cout << "[TEST] Measured q = " << q_val << ", p = " << p_val << std::endl;
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