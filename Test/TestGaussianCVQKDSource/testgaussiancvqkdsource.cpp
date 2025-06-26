/*!
 * \file
 * \brief Boost unit tests for quantum_gaussian_source
 *
 * Copyright (c) 2025 Aaron Abela
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>

#include "qkd/quantum_state.h"
#include "source/quantum_gaussian_source.h"
#include "truerand.h"

#include <iostream>
#include <memory>
#include <random>
#include <cmath>

using namespace libcomm;
using namespace libbase;
using namespace std;

BOOST_AUTO_TEST_CASE(test_single_gaussian_state) {
    // Create source with specific Gaussian parameters
    double q_mean_mean = 0.0;
    double q_mean_stddev = 10.0;
    double p_mean_mean = 0.0;
    double p_mean_stddev = 10.0;
    double q_stddev = 1.0;
    double p_stddev = 1.0;

    quantum_gaussian_source source(q_mean_mean, q_mean_stddev, p_mean_mean, p_mean_stddev, q_stddev, p_stddev);

    randgen r;
    r.seed(0);
    source.seedfrom(r);

    gaussian_state state = source.generate_single();

    std::cout << "\n[Generated a Single Gaussian State]" << std::endl;
    std::cout << "q_mean = " << state.get_q() << ", p_mean = " << state.get_p() << std::endl;

    BOOST_CHECK(std::isfinite(state.get_q()));
    BOOST_CHECK(std::isfinite(state.get_p()));
}

BOOST_AUTO_TEST_CASE(test_gaussian_sequence) {
    // Gaussian parameters
    double q_mean_mean = 0.0;
    double q_mean_stddev = 10.0;
    double p_mean_mean = 0.0;
    double p_mean_stddev = 10.0;
    double q_stddev = 1.0;
    double p_stddev = 1.0;

    quantum_gaussian_source source(q_mean_mean, q_mean_stddev, p_mean_mean, p_mean_stddev, q_stddev, p_stddev);

    randgen r;
    r.seed(7896);
    source.seedfrom(r);

    const int num_states = 10;
    libbase::size_type<libbase::vector> blocksize(num_states);
    libbase::vector<gaussian_state> sequence = source.generate_sequence(blocksize);

    std::cout << "\n[Generated Sequence of Gaussian States]" << std::endl;

    BOOST_CHECK_EQUAL(sequence.size(), num_states);

    for (int i = 0; i < sequence.size(); ++i) {
        gaussian_state& state = sequence(i);
        std::cout << "State: " << i
                  << ", q = " << state.get_q()
                  << ", p = " << state.get_p() << std::endl;

        BOOST_CHECK(std::isfinite(state.get_q()));
        BOOST_CHECK(std::isfinite(state.get_p()));
    }
}
