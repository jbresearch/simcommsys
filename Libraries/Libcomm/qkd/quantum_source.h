/*!
 * \file Source for generating quantum states for simulation of QKD protocols
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

#ifndef __quantum_source_h
#define __quantum_source_h

#include "qkd/quantum_state.h"

#include "vector.h"

#include <string>
#include <type_traits>

namespace libcomm
{

template <typename S>
class quantum_source
{
public:
    virtual S generate_state() = 0;
    virtual libbase::vector<S> generate_frame(int n)
    {
        libbase::vector<S> frame;
        frame.init(n);
        for (int i = 0; i < n; i++)
            frame(i) = generate_state();
        return frame;
    }
};

} // end namespace libcomm
#endif // __quantum_source_h