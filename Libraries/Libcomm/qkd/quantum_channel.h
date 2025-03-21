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

#ifndef __quantum_channel_h
#define __quantum_channel_h

#include "parametric.h"
#include "qkd/quantum_state.h"

#include <type_traits>

namespace libcomm
{

/*!
 * \brief   Common Base for Quantum channel.
 * \author  Mark Mizzi
 *
 * The class is parametrized by \name S_I, \name S_O, which represent input and
 * output quantum states respectively. The reason why we allow for different
 * types for input and output here is to allow the possibility of representing
 * an entangled input state, that is then separated in the output.
 */
template <
    typename S_I,
    typename S_O,
    typename T_I = typename S_I::measurement_type,
    typename T_O = typename S_O::measurement_type,
    std::enable_if_t<
        std::is_floating_point<std::is_base_of<quantum_state<T_I>, S_I>>::value,
        bool> = true,
    std::enable_if_t<
        std::is_floating_point<std::is_base_of<quantum_state<T_O>, S_O>>::value,
        bool> = true>
class quantum_channel : public parametric
{
public:
    virtual void set_input(S_I input) = 0;
    virtual S_O get_alice() = 0;
    virtual S_O get_bob() = 0;
};

} // end namespace libcomm
#endif // __quantum_channel_h