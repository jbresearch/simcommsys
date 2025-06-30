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
#include "qkd/quantum_channel.h"
#include "random.h"
#include "truerand.h"

#include <iostream>
#include <memory>
#include <cmath>

using namespace libcomm;
using namespace libbase;

BOOST_AUTO_TEST_CASE(test_single_gaussian_state) {
    quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
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
    quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
    randgen r;
    r.seed(7896);
    source.seedfrom(r);

    const int num_states = 10;
    vector<gaussian_state> sequence = source.generate_sequence(size_type<vector>(num_states));

    std::cout << "\n[Generated Sequence of Gaussian States]" << std::endl;
    BOOST_CHECK_EQUAL(sequence.size(), num_states);

    for (int i = 0; i < sequence.size(); ++i) {
        std::cout << "State " << i << ": q = " << sequence(i).get_q()
                  << ", p = " << sequence(i).get_p() << std::endl;

        BOOST_CHECK(std::isfinite(sequence(i).get_q()));
        BOOST_CHECK(std::isfinite(sequence(i).get_p()));
    }
}

BOOST_AUTO_TEST_CASE(test_gaussian_measurement_with_position_observable) {
    quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
    randgen r;
    r.seed(42);
    source.seedfrom(r);

    gaussian_state state = source.generate_single();

    double test_noise = 0.42;
    position_observable obs;
    obs.set_noise(test_noise);

    double q_noisy = obs.measure(state); // measured value

    std::cout << "\n[Position Observable Measurement]" << std::endl;
    std::cout << "Measured value (q + noise) = " << q_noisy << std::endl;

    BOOST_CHECK(std::isfinite(q_noisy));
}

BOOST_AUTO_TEST_CASE(test_gaussian_measurement_with_momentum_observable) {
    quantum_gaussian_source source(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
    randgen r;
    r.seed(84);
    source.seedfrom(r);

    gaussian_state state = source.generate_single();

    double test_noise = 0.57;
    momentum_observable obs;
    obs.set_noise(test_noise);

    double p_noisy = obs.measure(state); // measured value

    std::cout << "\n[Momentum Observable Measurement]" << std::endl;
    std::cout << "Measured value (p + noise) = " << p_noisy << std::endl;

    BOOST_CHECK(std::isfinite(p_noisy));
}

BOOST_AUTO_TEST_CASE(test_cvqkd_protocol_bob_observables_decision_vector) {
    cvqkd_protocol cvqkdprotocol;
    randgen rng;
    rng.seed(12345);
    cvqkdprotocol.seedfrom(rng);

    const int framesize = 10;
    std::vector<std::unique_ptr<observable<double>>> bob_observables = cvqkdprotocol.get_bob_observables(framesize);
    const libbase::vector<int>& decision_vector = cvqkdprotocol.get_decision_vector();

    BOOST_CHECK_EQUAL(bob_observables.size(), framesize);
    BOOST_CHECK_EQUAL(decision_vector.size(), framesize);

    std::cout << "\n[CVQKD Bob Observables and Decision Vector]" << std::endl;
    for (int i = 0; i < framesize; ++i) {
        std::string type = (decision_vector(i) == 0) ? "Position" :
                           (decision_vector(i) == 1) ? "Momentum" : "Unknown";

        std::cout << "Observable " << i << ": " << type << std::endl;

        if (decision_vector(i) == 0) {
            BOOST_CHECK(dynamic_cast<position_observable*>(bob_observables[i].get()) != nullptr);
        } else if (decision_vector(i) == 1) {
            BOOST_CHECK(dynamic_cast<momentum_observable*>(bob_observables[i].get()) != nullptr);
        } else {
            BOOST_FAIL("Invalid value in decision vector.");
        }
    }
}

BOOST_AUTO_TEST_CASE(test_cvqkd_protocol_bob_observables_decision_vector_using_pointer) {
    std::unique_ptr<cvqkd_protocol> protocol = std::make_unique<cvqkd_protocol>();
    std::cout<< "\nThis test uses pointer and pointer referencing " << std::endl;
    randgen rng;
    rng.seed(1237);
    protocol->seedfrom(rng);

    const int framesize = 10;
    std::vector<std::unique_ptr<observable<double>>> bob_observables = protocol->get_bob_observables(framesize);
    const libbase::vector<int>& decision_vector = protocol->get_decision_vector();

    BOOST_CHECK_EQUAL(bob_observables.size(), framesize);
    BOOST_CHECK_EQUAL(decision_vector.size(), framesize);

    std::cout << "\n[CVQKD Bob Observables and Decision Vector]" << std::endl;
    for (int i = 0; i < framesize; ++i) {
        std::string type = (decision_vector(i) == 0) ? "Position" :
                           (decision_vector(i) == 1) ? "Momentum" : "Unknown";

        std::cout << "Observable " << i << ": " << type << std::endl;

        if (decision_vector(i) == 0) {
            BOOST_CHECK(dynamic_cast<position_observable*>(bob_observables[i].get()) != nullptr);
        } else if (decision_vector(i) == 1) {
            BOOST_CHECK(dynamic_cast<momentum_observable*>(bob_observables[i].get()) != nullptr);
        } else {
            BOOST_FAIL("Invalid value in decision vector.");
        }
    }
}
