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
#include "qkd/observable.h"
#include "randgen.h"
#include "random.h"

#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>

namespace libcomm
{

//! \brief Reduced Planck constant
static constexpr double hbar = 1.054571817e-34;

/*!
 * \brief   Common Base for (non-entangled) quantum states.
 * \author  Mark Mizzi
 *
 * The template parameter \name S should be initialized to the concrete subclass
 * of this interface. It is used to cast *this to the right type when calling an
 * observable's measure() method.
 */
template <typename S>
class quantum_state_inf
{
public:
    //! \brief Used at a level that handles abstract quantum states to determine
    //! if state is entangled or not.
    static constexpr bool is_entangled = false;
    //! \brief Implements the other side of the visitor pattern, which calls the
    //! right measure() method of observable.
    template <typename T>
    T measure(observable<T>& o)
    {
        return o.measure(static_cast<S&>(*this));
    }
    virtual ~quantum_state_inf() {}
};

/*!
 * \brief   Common Base for entangled quantum states.
 * \author  Mark Mizzi
 *
 * The template parameter \name S should be initialized to the concrete subclass
 * of this interface. It is used to cast *this to the right type when calling an
 * observable's measure() method.
 */
template <typename S>
class entangled_quantum_state_inf
{
public:
    //! \brief Used at a level that handles abstract quantum states to determine
    //! if state is entangled or not.
    static constexpr bool is_entangled = true;
    //! \brief Implements the other side of the visitor pattern, which calls the
    //! right measure() method of observable.
    template <typename T>
    T measure(observable<T>& o, int idx)
    {
        return o.measure(static_cast<S&>(*this), idx);
    }
    virtual ~entangled_quantum_state_inf() {}
};

class qubit : quantum_state_inf<qubit>
{
private:
    std::complex<double> comp_basis_0, comp_basis_1;

public:
    qubit(std::complex<double> comp_basis_0, std::complex<double> comp_basis_1)
    {
        init(comp_basis_0, comp_basis_1);
    }

    void init(std::complex<double> comp_basis_0,
              std::complex<double> comp_basis_1)
    {
        // always ensure coefficients are normalized.
        assertalways(comp_basis_0 * std::conj(comp_basis_0) +
                         comp_basis_1 * std::conj(comp_basis_1) ==
                     std::complex<double>(1));
        this->comp_basis_0 = comp_basis_0;
        this->comp_basis_1 = comp_basis_1;
    }

    std::complex<double> get_comp_basis_0() const { return comp_basis_0; }
    std::complex<double> get_comp_basis_1() const { return comp_basis_1; }
};

// To confirm with Mark that the addition for public is okay.
class gaussian_state : public quantum_state_inf<gaussian_state>
{
private:
    std::mt19937 gen; //  Mersenne Twister random number generator
    bool was_measured = false; // Enforces measurement to be done only once for each created coherent state

    double q_mean, q_stddev, p_mean, p_stddev;

public:

    gaussian_state()
    : gen(), q_mean(0.0), q_stddev(1.0), p_mean(0.0), p_stddev(1.0) {}

    gaussian_state(double q_mean,
                   double q_stddev,
                   double p_mean,
                   double p_stddev)
                   : gen()
    {
        init(q_mean, q_stddev, p_mean, p_stddev);
    }


    void init(double q_mean, double q_stddev, double p_mean, double p_stddev)
    {
        assert(q_mean >= 0);
        assert(q_stddev >= 0);
        assert(p_mean >= 0);
        assert(p_stddev >= 0);
        assertalways(q_stddev * p_stddev >= hbar / 2);

        this->q_mean = q_mean;
        this->q_stddev = q_stddev;
        this->p_mean = p_mean;
        this->p_stddev = p_stddev;
    }

    //! Seeds the Mersenne Twister random number generator from a pseudo-random sequence
    void seedfrom(libbase::random& r)
    {
        gen.seed(r.ival());
    }

    // The two get functions are to be used for Measurement i.e. to implement the measurement in "observable.h" which is then used in "qkd_commsys.h". In both get_p and get_q, p_stddev = 1 and q_stddev = 1 respectively to get the measured values.
    double get_p()
    {
        assertalways(!was_measured && "Gaussian state can only be measured once.");
        was_measured = true;
        std::normal_distribution normdist{p_mean, p_stddev};
        return normdist(gen);
    }
    double get_q()
    {
        assertalways(!was_measured && "Gaussian state can only be measured once.");
        was_measured = true;
        std::normal_distribution normdist{q_mean, q_stddev};
        return normdist(gen);
    }
};

class entangled_qubit_pair : entangled_quantum_state_inf<entangled_qubit_pair>
{
};

class epr_beam : entangled_quantum_state_inf<epr_beam>
{
};

} // end namespace libcomm
#endif // __quantum_state_h