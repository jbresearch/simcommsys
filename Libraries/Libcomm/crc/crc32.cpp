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

#include "crc32.h"
using libbase::serializer;

namespace libcomm
{

// Registrar (matches the static member declared by the macro)
template <template <class> class C>
const serializer crc32_ieee<C>::shelper("crc", "crc_32", crc32_ieee<C>::create);

// Define the serialize functions declared by the macro
template <template <class> class C>
std::ostream&
crc32_ieee<C>::serialize(std::ostream& sout) const
{
    return sout;
}

template <template <class> class C>
std::istream&
crc32_ieee<C>::serialize(std::istream& sin)
{
    return sin;
}

// Explicit instantiation for libbase::vector container.
template class crc32_ieee<libbase::vector>;

} // namespace libcomm
