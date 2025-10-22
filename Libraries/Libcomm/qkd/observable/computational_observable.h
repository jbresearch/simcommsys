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

/*!
 * \brief Real Computational (Z-Basis) Observable
 * \author Aaron Abela
 *
 * An observable that performs a probabilistic quantum measurement
 * on a qubit in the Z-Basis (Computational).
 *
 * Reference: https://quantum-education-modules.readthedocs.io/en/latest/introductory/qubits/measurement.html
 */

#ifndef __computational_observable_h
#define __computational_observable_h

#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"
#include "random.h"
#include <cmath>
#include <complex>
#include <stdexcept>

namespace libcomm
{
class qubit;
class entangled_qubit_pair;
class epr_beam;

class computational_observable : public observable<bool>
{
private:

    mutable libbase::randgen rng;

    // This member will store the noise parameter from the depolarizing quantum channel.
    double qber;

public:

    // Constructor
    computational_observable() : rng(), qber(0.0) {}

    void seedfrom(libbase::random& r)
    {
        this->rng.seed(r.ival());
    }

    // Noise is added from the depolarising quantum channel from the transmit method.
    void set_qber(double qber_val) { qber = qber_val; }

    bool measure(qubit& state) const override
    {
        // Perform Ideal, Noiseless Measurement

        /* A qubit in the state |ψ⟩ = α|0⟩ + β|1⟩
        This represents a quantum superposition where α and β are complex amplitudes.*/
        std::complex<double> alpha = state.get_comp_basis_0();

        /* The probability of measuring |0⟩ is |α|^2. This is computed using the std::norm.*/
        double prob_0 = std::norm(alpha);

        /* To simulate a measurement, generate a uniform random number between 0 and 1.
        If the number is less than |α|^2, return 0 (i.e., collapse to |0⟩); else, return 1 (collapse to |1⟩).
        This follows the Born rule in simulation form.*/

        bool ideal_result;
        if (rng.fval_halfopen() < prob_0) {
            ideal_result = false; // Ideal result is 0
        } else {
            ideal_result = true;  // Ideal result is 1
        }

        // Apply Channel Noise (QBER)
        // Get a new random number to check for a bit-flip error.
        if (rng.fval_halfopen() < qber) {
            return !ideal_result; // A bit-flip error occurs
        } else {
            return ideal_result;  // No error
        }
    }

    void transmit(quantum_channel& c) override { c.transmit(*this); }

      bool measure(gaussian_state&) const override
    {
        failwith(
            "computational_observable does not support gaussian_state measurement.");
        return 0;
    }

    bool measure(entangled_qubit_pair&, int) const override
    {
        failwith("computational_observable does not support entangled qubit "
                 "pair measurement.");
        return 0;
    }

    bool measure(epr_beam&, int) const override
    {
        failwith(
            "computational_observable does not support EPR beam measurement.");
        return 0;
    }
};
}

#endif // __computational_observable_h