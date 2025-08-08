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

#ifndef __randgen_h
#define __randgen_h

#include "config.h"
#include "random.h"

namespace libbase
{

/*!
 * \brief   Knuth's Subtractive Random Generator.
 * \author  Johann Briffa
 *
 * A pseudo-random generator using the subtractive technique due to
 * Knuth. This algorithm was found to give very good results in the
 * communications lab during the third year.
 *
 * \note
 * - The subtractive algorithm has a very long period (necessary for low
 * bit error rates in the tested data stream)
 * - It also does not suffer from low-order correlations (facilitating its
 * use with a variable number of bits/code in the data stream)
 */

class randgen : public random
{
private:
    /*! \name Object representation */
    static constexpr int32_t mbig = 1000000000L;
    static constexpr int32_t mseed = 161803398L;
    int32_t next, nextp;
    int32_t ma[56], mj;
    // @}

protected:
    // Interface with random
#ifdef __CUDACC__
    __device__
    __host__
#endif
    void init(uint32_t s)
    {
        next = 0L;
        nextp = 31L;
        mj = (mseed - s) % mbig;
        ma[55] = mj;
        int32_t mk = 1;

        for (int i = 1; i <= 54; i++) {
            int ii = (21 * i) % 55;
            ma[ii] = mk;
            mk = mj - mk;
            if (mk < 0) {
                mk += mbig;
            }
            mj = ma[ii];
        }

        for (int k = 1; k <= 4; k++) {
            for (int i = 1; i <= 54; i++) {
                ma[i] -= ma[1 + (i + 30) % 55];
                if (ma[i] < 0) {
                    ma[i] += mbig;
                }
            }
        }
    }

#ifdef __CUDACC__
    __device__
    __host__
#endif
    void advance()
    {
        if (++next >= 56) {
            next = 1;
        }

        if (++nextp >= 56) {
            nextp = 1;
        }

        mj = ma[next] - ma[nextp];

        if (mj < 0) {
            mj += mbig;
        }

        ma[next] = mj;
    }

#ifdef __CUDACC__
    __device__
    __host__
#endif
    uint32_t get_value() const { return mj; }
#ifdef __CUDACC__
    __device__
    __host__
#endif
    uint32_t get_max() const { return mbig; }
};

} // namespace libbase

#endif
