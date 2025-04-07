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

} // end namespace libcomm