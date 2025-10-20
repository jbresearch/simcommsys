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
 * \brief Fake Computational (Z-Basis) Observable (Noiseless "Cheat")
 * \author Aaron Abela
 *
 * An observable that "measures" a BB84 qubit by reverse-engineering
 * its internal amplitudes to find the original BIT Alice used,
 * *assuming* she prepared it in the Z-Basis (Computational).
 *
 * This is for simulation/testing only. It will fail if used
 * on a state prepared in the X-Basis (Hadamard).
 */

#ifndef __fake_computational_observable_h
#define __fake_computational_observable_h

#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"
#include <cmath>
#include <complex>
#include <stdexcept> 

namespace libcomm
{

class qubit;
class entangled_qubit_pair;
class epr_beam;

class fake_computational_observable : public observable<bool>
{
public:
    fake_computational_observable() {}

    /*!
     * \brief "Measures" the qubit by deducing its Z-Basis bit.
     * \return `bool` representing the original bit (0 or 1).
     * \throws std::runtime_error if the state is not a Z-Basis state.
     */
    bool measure(qubit& state) const override
    {
        const double epsilon = 1e-9; // A small tolerance for float comparison

        // Get the internal amplitudes. 
        std::complex<double> alpha = state.get_comp_basis_0();
        std::complex<double> beta = state.get_comp_basis_1();

        // Case 1: State |0> (bit=0, basis=0)
        // alpha=1.0, beta=0.0
        if (std::abs(alpha.real() - 1.0) < epsilon && std::abs(beta.real()) < epsilon) {
            return false; // bit=0
        }

        // Case 2: State |1> (bit=1, basis=0)
        // alpha=0.0, beta=1.0
        if (std::abs(alpha.real()) < epsilon && std::abs(beta.real() - 1.0) < epsilon) {
            return true; // bit=1
        }

        // If it's none of these, it's not a Z-basis state (e.g., |+> or |->)
        throw std::runtime_error(
            "FakeComputationalObservable: State is not a Z-Basis state.");
    }

    void transmit(quantum_channel& c) override { c.transmit(*this); }

    bool measure(gaussian_state&) const override
    {
        failwith(
            "fake_computational_observable does not support gaussian_state measurement.");
        return 0;
    }

    bool measure(entangled_qubit_pair&, int) const override
    {
        failwith("fake_computational_observable does not support entangled qubit "
                 "pair measurement.");
        return 0;
    }

    bool measure(epr_beam&, int) const override
    {
        failwith(
            "fake_computational_observable does not support EPR beam measurement.");
        return 0;
    }
};


} // namespace libcomm

#endif // __fake_computational_observable_h