/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#ifndef __rand_lut_h
#define __rand_lut_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "randgen.h"
#include "serializer.h"
#include <iostream>

namespace libcomm
{

/*!
 * \brief   Random LUT Interleaver.
 * \author  Johann Briffa
 *
 * \note This assumes the implementation of a random simile interleaver; there
 * is therefore a restriction that the interleaver size must be a
 * multiple of p, where p is the length of the encoder impulse response
 * (cf my MPhil p.40). The constructor gives an error if this is not the
 * case.
 */

template <class real>
class rand_lut : public lut_interleaver<real>
{
    int p;
    libbase::randgen r;

protected:
    void init(const int tau, const int m);
    rand_lut() {}

public:
    rand_lut(const int tau, const int m) { init(tau, m); }
    ~rand_lut() {}

    void seedfrom(libbase::random& r);
    void advance();

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(rand_lut)
};

} // namespace libcomm

#endif
