/*!
 * \file
 *
 * Copyright (c) 2025 Johann A. Briffa
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

#ifndef __sign_h
#define __sign_h

#include "config.h"
#include "informed_embedder.h"
#include "randgen.h"

namespace libcomm
{

/*!
 * \brief   Sign Embedder/Extractor.
 * \author  Johann Briffa
 *
 * This class implements sign embedding that can be applied to any floating
 * point type.
 */

template <class S>
class sign : public informed_embedder<S>
{
public:
    sign() {}

    // Atomic informed_embedder operations
    const S embed(const int i, const S s) const
    {
        // sign embedding works only for binary sources
        assert(i >= 0 && i < 2);
        // sign change represents 1, no sign change represents 0
        if (i == 0) {
            return s;
        } else {
            return -s;
        }
    }
    const int extract(const S& rx, const S& reference) const
    {
        if (std::signbit(rx) == std::signbit(reference)) {
            return 0;
        } else {
            return 1;
        }
    }

    // Informative functions
    int num_symbols() const { return 2; }

    // Description
    std::string description() const { return "Sign embedder"; };

    // Serialization Support
    DECLARE_SERIALIZER(sign)
};

} // namespace libcomm

#endif
