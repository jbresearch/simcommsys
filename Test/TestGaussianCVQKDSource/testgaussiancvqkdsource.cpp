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
#include <boost/test/included/unit_test.hpp>

#include "qkd/quantum_state.h"
#include "source/quantum_gaussian_source.h"
#include "source.h"
#include "serializer.h"
#include "truerand.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>

using namespace libcomm;
using namespace libbase;
using namespace std;

BOOST_AUTO_TEST_CASE(print_single_gaussian_state) {
    libbase::vector<double> params;
    params.init(6);
    params(0) = 0.0;   // q_mean_mean
    params(1) = 10.0;  // q_mean_stddev
    params(2) = 0.0;   // p_mean_mean
    params(3) = 10.0;  // p_mean_stddev
    params(4) = 1.0;   // q_stddev
    params(5) = 1.0;   // p_stddev

    quantum_gaussian_source source;
    source.set_parameters(params);

    randgen r;
    r.seed(0);
    source.seedfrom(r);

    std::cout << "\n[Generated a Single Gaussian State]" << std::endl;
    gaussian_state state = source.generate_single();
    // In case you want to print the q_mean and p_mean of each state make sure to uncomment lines 76-76 in the generate_single method found in quantum_gaussian_source.cpp.
}

BOOST_AUTO_TEST_CASE(print_gaussian_sequence) {
    libbase::vector<double> params;
    params.init(6);
    params(0) = 0.0;
    params(1) = 10.0;  // q_mean_stddev
    params(2) = 0.0;
    params(3) = 10.0;  // p_mean_stddev
    params(4) = 1.0;   // q_stddev
    params(5) = 1.0;   // p_stddev

    quantum_gaussian_source source;
    source.set_parameters(params);

    randgen r;
    r.seed(7896);
    source.seedfrom(r);

    const int num_states = 10;
    libbase::size_type<libbase::vector> blocksize(num_states);
    libbase::vector<gaussian_state> sequence = source.generate_sequence(blocksize);

    std::cout << "\n[Generated Sequence of Gaussian States]" << std::endl;
    for (int i = 0; i < sequence.size(); ++i) {
        gaussian_state& state = sequence(i);
        cout << "State: " << i << endl;
        cout << "p = " << state.get_p() << endl;
        cout << "q = " << state.get_q() << endl;
    }
}
