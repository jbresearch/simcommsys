/*!
 * \file
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

#include "qkd/quantum_channel.h"
#include "serializer.h"

#include <iostream>
#include <memory>


using libbase::serializer;

namespace libcomm
{

// Description

std::string
identity_quantum_channel::description() const
{
    return "Identity quantum channel";
}

// Serialization Support

std::ostream&
identity_quantum_channel::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
identity_quantum_channel::serialize(std::istream& sin)
{
    return sin;
}

// Added missing shelper for Identity Quantum Channel
const serializer identity_quantum_channel::shelper(
    "quantum_channel", "identity_quantum_channel", identity_quantum_channel::create);


/*!
 * \brief   Gaussian Quantum channel serialization support
 * \author  Aaron Abela
 *
 * Noise is modelled as a Normal Distribution with Mean zero and Variance V_N:
 *     V_N = N_0 + ηTξ + v_el
 * Bob's measured value:
 *     X_B = sqrt(ηT) * (X_A + X_N)
 */

// Description for Gaussian Quantum Channel
std::string gaussian_quantum_channel::description() const {
        return "Gaussian quantum channel model for GM Coherent States CV-QKD";
    }

// Serialization support
std::ostream& gaussian_quantum_channel::serialize(std::ostream& sout) const
 {
     sout << "# Vaccuum/Shot Noise (N_0)" << std::endl;
     sout << N_0 << std::endl;
     sout << "#  Attenuation coefficient (Alpha)" << std::endl;
     sout << alpha << std::endl;
     sout << "# Detector Efficiency (eta)" << std::endl;
     sout << det_eff << std::endl;
     sout << "Distance in Km (l)" << std::endl;
     sout << distance_km << std::endl;
     sout << "Excess Noise (xi)" << std::endl;
     sout << excess_noise << std::endl;
     sout << "Electric Noise (v_el)" << std::endl;
     sout << electric_noise << std::endl;
     return sout;
 }

 std::istream& gaussian_quantum_channel::serialize(std::istream& sin)
 {
     assertalways(sin.good());
     sin >> libbase::eatcomments >> N_0;
     sin >> libbase::eatcomments >> alpha;
     sin >> libbase::eatcomments >> det_eff;
     sin >> libbase::eatcomments >> distance_km;
     sin >> libbase::eatcomments >> excess_noise;
     sin >> libbase::eatcomments >> electric_noise;
     return sin;
 }

// Register with serializer system
const serializer gaussian_quantum_channel::shelper(
    "quantum_channel", "gaussian_quantum_channel", gaussian_quantum_channel::create);

} // end namespace libcomm