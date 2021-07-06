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

#ifndef __helical_h
#define __helical_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"

namespace libcomm
{

/*!
 * \brief   Helical Interleaver.
 * \author  Johann Briffa
 *
 */

template <class real>
class helical : public lut_interleaver<real>
{
    int rows, cols;

protected:
    void init(const int tau, const int rows, const int cols);
    helical() {}

public:
    helical(const int tau, const int rows, const int cols)
    {
        init(tau, rows, cols);
    }
    ~helical() {}

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(helical)
};

} // namespace libcomm

#endif
