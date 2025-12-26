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
 * TODO: rename to quantum_coherent_source
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __quantum_gaussian_source_h
#define __quantum_gaussian_source_h

#include "config.h"
#include "qkd/quantum_state.h"
#include "serializer.h"
#include "source.h"

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

namespace libcomm
{

/*!
 * \brief   Gaussian state source (not-templated)
 * \author  Aaron Abela
 *
 * Implements a source for CV-QKD using the GG02 protocol that returns a
 Gaussian coherent state with q_mean and p_mean, where p_mean and q_mean follow
 a normal distribution with p_mean_mean, p_mean_stddev, q_mean_mean and
 q_mean_stddev respectively. Note: Variance = (Stddev)^2 and Q and P are the
 quadrature components of the coherent state.

 Inputs: q_mean_mean, q_mean_stddev, p_mean_mean, p_mean_stddev, q_stddev,
 p_stddev Return: Coherent state with q_mean, p_mean, q_stddev and p_stddev
 *
 * TODO: rename to quantum_coherent_source
 */

class quantum_gaussian_source : public source<gaussian_state, libbase::vector>
{

private:
    std::mt19937 gen;
    double q_mean_mean;   // Chosen mean to generate q_mean
    double q_mean_stddev; // Chosen stddev to generate q_mean
    double p_mean_mean;   // Chosen mean to generate p_mean
    double p_mean_stddev; // Chosen stddev to generate p_mean
    double q_stddev;      // Stddev of q
    double p_stddev;      // Stddev of p

public:
    // Default constructor
    quantum_gaussian_source(double q_mean_mean = 0.0,
                            double q_mean_stddev = 1.0,
                            double p_mean_mean = 0.0,
                            double p_mean_stddev = 1.0,
                            double q_stddev = 1.0,
                            double p_stddev = 1.0)
        : q_mean_mean(q_mean_mean), q_mean_stddev(q_mean_stddev),
          p_mean_mean(p_mean_mean), p_mean_stddev(p_mean_stddev),
          q_stddev(q_stddev), p_stddev(p_stddev)
    {
    }

    //! Generate a single Gaussian state with q_mean and p_mean.
    gaussian_state generate_single() override
    {
        std::normal_distribution<double> q_dist(q_mean_mean, q_mean_stddev);
        std::normal_distribution<double> p_dist(p_mean_mean, p_mean_stddev);
        double q_mean = q_dist(gen); // Value will have added noise to it to be
                                     // used for measurement.
        double p_mean = p_dist(gen); // Value will have added noise to it to be
                                     // used fWor measurement.
        return gaussian_state(q_mean, q_stddev, p_mean, p_stddev);
    }

    //! Seeds the Mersenne Twister random number generator from a pseudo-random
    //! sequence
    void seedfrom(libbase::random& r) override { gen.seed(r.ival()); }

    double get_VA() const
    {
        const double VA = ((q_mean_stddev * q_mean_stddev) +
              (p_mean_stddev * p_mean_stddev)) /
             2;
        return VA;
    }

    // Description
    std::string description() const;

    // Helper function - Required only for TestGaussianCVQKDsource with Boost
    // usage
    static std::unique_ptr<libbase::serializable> create(std::istream& sin)
    {
        auto obj = std::make_unique<quantum_gaussian_source>();
        obj->serialize(sin);
        return obj;
    }

    // Serialization Support
    DECLARE_SERIALIZER(quantum_gaussian_source)
};

} // namespace libcomm

#endif // __quantum_gaussian_source_h
