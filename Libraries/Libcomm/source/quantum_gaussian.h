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

#ifndef __source_quantum_gaussian_h
#define __source_quantum_gaussian_h

#include "config.h"
#include "serializer.h"
#include "source.h"
#include "qkd/quantum_state.h"

#include <random>
#include <string>
#include <type_traits>
#include <sstream>
#include <memory>

namespace libcomm {

/*!
 * \brief   Gaussian state source (not-templated)
 * \author  Aaron Abela
 *
 * Implements a source for CV-QKD using the GG02 protocol that returns a Gaussian coherent state with q_mean and p_mean, where p_mean and q_mean follow a normal distribution with mean_p_mean, stddev_p_mean, mean_q_mean and stddev_q_mean respectively. Note: Variance = (Stddev)^2 and Q and P are the quadrature components of the coherent state.

 Inputs: mean_q_mean, stddev_q_mean, mean_p_mean, stddev_p_mean
 Return: Coherent state with q_mean and p_mean
 *
 *
 */


class quantum_gaussian: public source<gaussian_state, libbase::vector> {

private:
    double mean_q_mean; // Chosen mean to generate q_mean
    double stddev_q_mean; // Chosen stddev to generate q_mean
    double mean_p_mean; // Chosen mean to generate p_mean
    double stddev_p_mean;  // Chosen stddev to generate p_mean
    std::mt19937 gen;

public:
    // Default constructor
    quantum_gaussian(double mean_q_mean = 0.0, double stddev_q_mean = 1.0, double mean_p_mean = 0.0, double stddev_p_mean = 1.0)
        : mean_q_mean(mean_q_mean), stddev_q_mean(stddev_q_mean), mean_p_mean(mean_p_mean), stddev_p_mean(stddev_p_mean), gen(std::random_device{}()) {}

    //! Generate a single Gaussian state with q_mean and p_mean.
    gaussian_state generate_single() override {
        std::normal_distribution<double> q_dist(mean_q_mean, stddev_q_mean);
        std::normal_distribution<double> p_dist(mean_p_mean, stddev_p_mean);
        double q_mean = q_dist(gen); // Value will have added noise to it to be used for measurement.
        double p_mean = p_dist(gen); // Value will have added noise to it to be used for measurement.
        return gaussian_state(q_mean, stddev_q_mean, p_mean, stddev_p_mean);
    }

    // Sets the seed directly using the std::mt19937
    void set_seed(unsigned int seed) {
        gen.seed(seed);
    }

    // Required for TestGaussianCVQKDsource with Boost usage
    static std::unique_ptr<libbase::serializable> create(std::istream& sin) {
        auto obj = std::make_unique<quantum_gaussian>();
        obj->serialize(sin);
        return obj;
    }

    //! Description
    std::string description() const override {
        std::ostringstream sout;
        sout << "Quantum Gaussian random source ("
        << "q_mean ~ N(" << mean_q_mean << ", " << stddev_q_mean << "), "
        << "p_mean ~ N(" << mean_p_mean << ", " << stddev_p_mean << "))";
        return sout.str();
    }

    // Serialization Support
    DECLARE_SERIALIZER(quantum_gaussian)
};

} // namespace libcomm

#endif // __source_quantum_gaussian_h
