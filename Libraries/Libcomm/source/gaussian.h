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

#ifndef __source_gaussian_h
#define __source_gaussian_h

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
 * \brief   Gaussian state source
 * \author  Aaron Abela
 *
 * Implements a source that returns a Gaussian coherent state which follows a normal distribution with mean zero and variance VA.
 */

template <class S = gaussian_state, template <class> class C = libbase::vector>
class gaussian : public source<S, C> {
    static_assert(std::is_same_v<S, gaussian_state>, "gaussian<> only supports gaussian_state");

private:
    double mean;
    double stddev;
    std::mt19937 gen;

public:
    // Default constructor
    gaussian(double mean = 0.0, double stddev = 1.0)
        : mean(mean), stddev(stddev), gen(std::random_device{}()) {}

    //! Generate a single Gaussian state with quadratures p and q
    S generate_single() override {
        std::normal_distribution<double> q_dist(mean, stddev);
        std::normal_distribution<double> p_dist(mean, stddev);
        double q = q_dist(gen);
        double p = p_dist(gen);
        return S(q, p);
    }

    //! Seeds any random generators from a pseudo-random sequence
    void seedfrom(libbase::random& r) override {
        gen.seed(r.ival());
    }

    // Sets the seed directly using the std::mt19937
    void set_seed(unsigned int seed) {
        gen.seed(seed);
    }

    // Required for TestGaussianCVQKDsource with Boost usage
    static std::unique_ptr<libbase::serializable> create(std::istream& sin) {
        auto obj = std::make_unique<gaussian<S, C>>();
        obj->serialize(sin);
        return obj;
    }

    //! Description
    std::string description() const override {
        std::ostringstream sout;
        sout << "Gaussian random source (mean=" << mean << ", stddev=" << stddev << ")";
        return sout.str();
    }

    // Serialization Support
    DECLARE_SERIALIZER(gaussian)
};

} // namespace libcomm

#endif // __source_gaussian_h
