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

#include "quantum_bb84_source.h"
#include <sstream>

using libbase::serializer;

namespace libcomm
{

//! Description
std::string
quantum_bb84_source::description() const
{
    return "Qubit Source for BB84.";
}

std::ostream&
quantum_bb84_source::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
quantum_bb84_source::serialize(std::istream& sin)
{
    return sin;
}

// Register with serializer system
const serializer quantum_bb84_source::shelper("source",
                                              "quantum_bb84_source",
                                              quantum_bb84_source::create);

} // namespace libcomm
