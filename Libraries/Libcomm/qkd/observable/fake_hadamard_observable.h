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
 * \brief Fake Hadamard Observable (Noiseless)
 * \author Aaron Abela
 *
* An observable that "measures" a BB84 qubit by reverse-engineering
 * its internal amplitudes to find the original BIT Alice used,
 * *assuming* she prepared it in the Z-Basis (Rectilinear or Computational).
 */

#ifndef __fake_hadamard_observable_h
#define __fake_hadamard_observable_h

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

class fake_hadamard_observable : public observable<bool>
{
public:
    fake_hadamard_observable() {}

    // Perfect measurement with no noise or loss.
    bool measure(qubit& state) const override
    {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        const double epsilon = 1e-9; // A small tolerance for float comparison. 

        // Get the internal amplitudes. 
        std::complex<double> alpha = state.get_comp_basis_0();
        std::complex<double> beta = state.get_comp_basis_1();

        // Case 3: State |+> (bit=0, basis=1)
        // alpha=1/sqrt(2), beta=1/sqrt(2)
        if (std::abs(alpha.real() - inv_sqrt2) < epsilon &&
            std::abs(beta.real() - inv_sqrt2) < epsilon) {
            return false; // bit=0
        }

        // Case 4: State |-> (bit=1, basis=1)
        // alpha=1/sqrt(2), beta=-1/sqrt(2)
        if (std::abs(alpha.real() - inv_sqrt2) < epsilon &&
            std::abs(beta.real() + inv_sqrt2) < epsilon) {
            return true; // bit=1
        }

        // If it's none of these, it's not an X-basis state (e.g., |0> or |1>)
        throw std::runtime_error(
            "FakeHadamardObservable: State is not an X-Basis state.");
    }

    void transmit(quantum_channel& c) override { c.transmit(*this); }

    bool measure(gaussian_state&) const override
    {
        failwith(
            "fake_hadamard_observable does not support gaussian_state measurement.");
        return 0;
    }

    bool measure(entangled_qubit_pair&, int) const override
    {
        failwith("fake_hadamard_observable does not support entangled qubit "
                 "pair measurement.");
        return 0;
    }

    bool measure(epr_beam&, int) const override
    {
        failwith(
            "fake_hadamard_observable does not support EPR beam measurement.");
        return 0;
    }
};


} // namespace libcomm

#endif // __fake_hadamard_observable_h