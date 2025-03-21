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

#ifndef __qkd_postprocessor_h
#define __qkd_postprocessor_h

#include "qkd/quantum_state.h"
#include "vector.h"

#include <type_traits>

namespace libcomm
{

/*!
 * \brief   Common Base for QKD postprocessing.
 * \author  Mark Mizzi
 */
template <typename S,
          typename T = typename S::measurement_type,
          std::enable_if_t<std::is_floating_point<
                               std::is_base_of<quantum_state<T>, S>>::value,
                           bool> = true>
class qkd_postprocessor
{
public:
    virtual void set_alice(libbase::vector<S> chan_output) = 0;
    virtual void set_bob(libbase::vector<S> chan_output) = 0;
    virtual libbase::vector<bool> get_secret_key() = 0;
};

} // end namespace libcomm

#endif // __qkd_postprocessor_h