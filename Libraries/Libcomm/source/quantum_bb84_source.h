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

#ifndef __quantum_bb84_source_h
#define __quantum_bb84_source_h

#include "config.h"
#include "qkd/quantum_state.h"
#include "serializer.h"
#include "source.h"

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <cmath>

/*!
 * \brief   Source for the BB84 QKD protocol.
 * \author  Aaron Abela
 *
 * Implements a source for the BB84 protocol. For each call to
 * generate_single(), it randomly chooses a bit (0 or 1) and a basis (Z or X),
 * and returns the corresponding qubit state. It internally stores both vectors
 * for later use.
 */

namespace libcomm
{
class quantum_bb84_source : public source<qubit, libbase::vector>
{
private:
    libbase::randgen rng;

    // Vectors to store Alice's information
    // TODO: To delete after testing is done. 
    std::vector<bool> alice_bits; // vector a
    std::vector<bool> alice_bases; // vector b

public:
    //! Default constructor
    quantum_bb84_source() {}

    //! Generate a single qubit for the BB84 protocol.
    qubit generate_single() override
    {
        // Generate a random bit and a random basis.
        bool bit = (rng.ival(2) != 0);
        bool basis = (rng.ival(2) != 0);

        alice_bits.push_back(bit);
        alice_bases.push_back(basis);

        // 3. Create the corresponding qubit state based on the choices.
        if (basis == 0) { // Z-basis (Rectilinear or Computational Basis)
            if (bit == 0) {
                // State |0>
                return qubit({1.0, 0.0}, {0.0, 0.0});
            } else {
                // State |1>
                return qubit({0.0, 0.0}, {1.0, 0.0});
            }
        } else { // X-basis (Diagonal or Hadamard Basis)
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            if (bit == 0) {
                // State |+>
                return qubit({inv_sqrt2, 0.0}, {inv_sqrt2, 0.0});
            } else {
                // State |->
                return qubit({inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0});
            }
        }
    }

    /* Getters for bit and basis vectors of Alice. These will only
    be used for testing purposes. 
    TODO: They need to be deleted. */
    const std::vector<bool>& get_bits() const { return alice_bits; }
    const std::vector<bool>& get_bases() const { return alice_bases; }

    //!  Seeds from libbase::randgen.
    void seedfrom(libbase::random& rng) override
    {
        this->rng.seed(rng.ival());
    }

    // Description
    std::string description() const;

    // static create method required by the serializer.
    // TODO: To check if I need to delete this. 
    static std::unique_ptr<libbase::serializable> create(std::istream& sin)
    {
        auto obj = std::make_unique<quantum_bb84_source>();
        obj->serialize(sin);
        return obj;
    }

    // Serialization Support
    DECLARE_SERIALIZER(quantum_bb84_source)

};

} // namespace libcomm

#endif // __quantum_bb84_source_h