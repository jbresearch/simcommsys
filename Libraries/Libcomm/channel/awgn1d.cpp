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

#include "awgn1d.h"

namespace libcomm
{


const libbase::serializer awgn1d::shelper("channel", "awgn1d", awgn1d::create);

// handle functions

void
awgn1d::compute_parameters(const double Eb_in, const double No_in)
{
        sigma = std::sqrt(Eb_in * No_in);
}

// channel handle functions

double
awgn1d::corrupt(const double& s)
{
    return s + this->r.gval(sigma);
}

double
awgn1d::pdf(const double& tx, const double& rx) const
{
    // libbase::gauss expects normalized arg
    return libbase::gauss((rx - tx) / sigma);
}

// Description

std::string
awgn1d::description() const
{
    return "AWGN channel (scalar)";
}

// Serialization Support

std::ostream& awgn1d::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream& awgn1d::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
