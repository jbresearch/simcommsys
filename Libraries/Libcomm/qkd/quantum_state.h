/*!
 * \file Representation of quantum states for simulation of QKD protocols
 *
 * Copyright (c) 2025 Mark Mizzi
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

#ifndef __quantum_state_h
#define __quantum_state_h

#include "assertalways.h"
#include "randgen.h"
#include "random.h"

#include <cmath>
#include <memory>
#include <string>

namespace libcomm
{

static constexpr double hbar = 1.054571817e-34;

/*!
 * \brief   Common Base for Quantum state.
 * \author  Mark Mizzi
 *
 * The class is parametrized by \name T which represents the type for a
 * measurement outcome (e.g. \name bool for a qubit). We represent measurement
 * of a state by a virtual method, but not transformations, as the latter are
 * too difficult to represent generally. Transformations should be represented
 * at the concrete subclass level.
 */
template <typename T>
class quantum_state
{
public:
    using measurement_type = T;

    /*! \brief Represents measurement of a named observable.
     *
     * For the purposes of QKD simulation, we can always use a single type to
     * represent all measurement outcomes, since e.g. measurement of either
     * quadrature in CV-QKD yields a double.
     */
    virtual T measure(std::string observable) = 0;
    virtual ~quantum_state() {}
};

class qubit : public quantum_state<bool>
{
private:
    std::unique_ptr<libbase::random> randgen;
    // Bloch sphere parameters for the state in the computational basis
    double bloch_sphere_theta_computational, bloch_sphere_phi_computational;

public:
    qubit(double bloch_sphere_theta_computational,
          double bloch_sphere_phi_computational,
          double seed,
          std::unique_ptr<libbase::random>&& randgen =
              std::make_unique<libbase::randgen>())
        : bloch_sphere_theta_computational(bloch_sphere_theta_computational),
          bloch_sphere_phi_computational(bloch_sphere_phi_computational),
          randgen(std::move(randgen))
    {
        // ensure that angles are valid for Bloch sphere
        assertalways(bloch_sphere_theta_computational <= M_PI);
        assertalways(bloch_sphere_phi_computational < 2 * M_PI);
        this->randgen->seed(seed);
    }

    bool measure(std::string observable) override
    {
        if (observable == "computational") {
            const double val = randgen->fval_halfopen();
            return val < pow(sin(bloch_sphere_theta_computational / 2), 2);
        } else if (observable == "hadamard") {
            const double val = randgen->fval_halfopen();
            return val < (0.5 + cos(bloch_sphere_phi_computational) *
                                    cos(bloch_sphere_theta_computational / 2) *
                                    sin(bloch_sphere_theta_computational / 2));
        } else {
            failwith(
                std::string("Attempted measure of unsupported observable ") +
                observable);
        }
    }
};

template <typename T>
class quantum_gaussian_state : public quantum_state<T>
{

protected:
    // Gaussian parameters
    double q_mean, q_stddev, p_mean, p_stddev;

public:
    quantum_gaussian_state(double q_mean,
                           double q_stddev,
                           double p_mean,
                           double p_stddev)
        : q_mean(q_mean), q_stddev(q_stddev), p_mean(p_mean), p_stddev(p_stddev)
    {
        // ensure that state obeys Heisenberg uncertainty principle
        assertalways(p_stddev * q_stddev >= hbar / 2);
    }
};

class quantum_gaussian_state_homodyne : public quantum_gaussian_state<double>
{
private:
    std::unique_ptr<libbase::random> randgen;

public:
    quantum_gaussian_state_homodyne(double q_mean,
                                    double q_stddev,
                                    double p_mean,
                                    double p_stddev,
                                    double seed,
                                    std::unique_ptr<libbase::random>&& randgen =
                                        std::make_unique<libbase::randgen>())
        : quantum_gaussian_state(q_mean, q_stddev, p_mean, p_stddev),
          randgen(std::move(randgen))
    {
        this->randgen->seed(seed);
    }

    double measure(std::string observable) override
    {
        if (observable == "p") {
            return randgen->gval(quantum_gaussian_state::p_stddev) +
                   quantum_gaussian_state::p_mean;
        } else if (observable == "q") {
            return randgen->gval(quantum_gaussian_state::q_stddev) +
                   quantum_gaussian_state::q_mean;
        } else {
            failwith(
                std::string("Attempted measure of unsupported observable ") +
                observable);
        }
    }
};

} // end namespace libcomm
#endif // __quantum_state_h