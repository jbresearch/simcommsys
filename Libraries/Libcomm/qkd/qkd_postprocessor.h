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

#include "instrumented.h"
#include "qkd/observable.h"
#include "vector.h"

#include <type_traits>

namespace libcomm
{

/*!
 * \brief   Common Base for QKD postprocessing.
 * \author  Mark Mizzi
 */
template <typename T>
class qkd_postprocessor : public instrumented
{
public:
    /*! Get observables used to measure quantum states on Alice's end, e.g. spin
     * in two different bases for E91
     * Integer param determines number of observables returned.
     */
    virtual libbase::vector <
        std::unique_ptr<observable<T>> get_alice_observables(int) = 0;
    /*! Get observables used to measure quantum states on Bob's end, e.g. spin
     * in two different bases for E91
     * Integer param determines number of observables returned.
     */
    virtual libbase::vector <
        std::unique_ptr<observable<T>> get_bob_observables(int) = 0;

    virtual libbase::vector<bool>
    postprocess(const libbase::vector<T>&& bob_measurements,
                const libbase::vector<T>&& alice_measurements) = 0;

    virtual ~qkd_postprocessor() {}
};

} // end namespace libcomm

#endif // __qkd_postprocessor_h