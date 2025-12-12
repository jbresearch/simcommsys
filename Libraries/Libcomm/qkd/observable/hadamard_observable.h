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
 * \brief Real Hadamard (X-Basis) Observable
 * \author Aaron Abela
 *
 * An observable that performs a probabilistic quantum measurement
 * on a qubit in the X-Basis (Hadamard).
 *
 * Reference: https://quantum-education-modules.readthedocs.io/en/latest/introductory/qubits/measurement.html
 */

#ifndef __hadamard_observable_h
#define __hadamard_observable_h

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

class hadamard_observable : public observable<bool>
{
private:
    // This member will store the noise parameter from the depolarizing quantum channel.
    double qber;

public:
    // Constructor
    hadamard_observable() : qber(0.0) {}

    // Noise is added from the depolarising quantum channel from the transmit method.
    void set_qber(double qber_val) { qber = qber_val; }

    bool measure(qubit& state) const override
    {

        /*
        To measure in the Hadamard basis (|+⟩, |−⟩), we first express the state in that basis:

        |ψ⟩ = α|0⟩ + β|1⟩ = c₊|+⟩ + c₋|−⟩

        Where:
        c₊ = ⟨+|ψ⟩ = (α + β) / √2
        c₋ = ⟨−|ψ⟩ = (α - β) / √2

        The probability of measuring |+⟩ is |c₊|²
        */

        // Perform Ideal, Noiseless measurement
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

        // Get Z-Basis amplitudes
        std::complex<double> alpha = state.get_comp_basis_0();
        std::complex<double> beta = state.get_comp_basis_1();

        // Convert to X-Basis amplitudes.
        // Only the amplitude for the |+> state (bit 0) is required.
        // alpha_X = (alpha + beta) / sqrt(2)
        std::complex<double> alpha_X = (alpha + beta) * inv_sqrt2;

        // Calculate probability of measuring |+> (bit 0)
        double prob_plus = std::norm(alpha_X);

        /* To simulate a measurement, generate a uniform random number between 0 and 1.
        If the number is less than prob_plus, return 0 (i.e., collapse to |+⟩); else, return 1 (collapse to |−⟩).
        This follows the Born rule in simulation form.
        */

        bool ideal_result;
        if (rng.fval_halfopen() < prob_plus) {
            ideal_result = false; // Ideal result is 0 (state |+>)
        } else {
            ideal_result = true;  // Ideal result is 1 (state |->)
        }

          // -------------------------------------------------------------------
        // Apply Channel Noise (Binary Symmetric Channel Model)
        // -------------------------------------------------------------------
        /*
         * SimCommSys models the quantum channel noise phenomenologically using 
         * a Binary Symmetric Channel (BSC) applied post-measurement.
         * * Instead of evolving the density matrix (Depolarizing Channel), we 
         * apply a probabilistic bit-flip to the classical measurement result.
         * * qber (Quantum Bit Error Rate) acts as the crossover probability 'epsilon':
         * - If rng < qber: An error occurs (Bit Flip: 0->1 or 1->0).
         * - If rng >= qber: The result remains correct.
         * * Note: A qber of 0.5 represents maximum entropy (random guessing),
         * whereas a qber of 1.0 represents a deterministic inversion (NOT gate).
         */


        // --- Apply Channel Noise (QBER) ---
        // Get a new random number to check for a bit-flip error.
        if (rng.fval_halfopen() < qber) {
            return !ideal_result; // A bit-flip error occurs, BIT FLIP (0->1 or 1->0)
        } else {
            return ideal_result;  // No error
        }
    }

    void transmit(quantum_channel& c) override { c.transmit(*this); }

    bool measure(gaussian_state&) const override
    {
        failwith(
            "hadamard_observable does not support gaussian_state measurement.");
        return 0;
    }

    bool measure(entangled_qubit_pair&, int) const override
    {
        failwith("hadamard_observable does not support entangled qubit "
                 "pair measurement.");
        return 0;
    }

    bool measure(epr_beam&, int) const override
    {
        failwith(
            "hadamard_observable does not support EPR beam measurement.");
        return 0;
    }
};
} // namespace libcomm

#endif // __hadamard_observable_h