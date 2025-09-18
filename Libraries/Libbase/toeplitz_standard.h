/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela.
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

/*!
 * Documentation of Code:
 *
 * This code is based on the Python implementation of the scipy.linalg.toeplitz
 * function. Reference:
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toe
 *
 * The Toeplitz matrix is built from a single starting vector.
 *
 * sv is the starting_vector which will be used to build the toeplitz matrix.
 *
 * L is the length of the final hashed key outputted after privacy
 * amplification.
 *
 * N is the length of the inputted key that will be hashed. In the case of MDR
 * the input vector is Bob's s vector of size k.
 *
 * Builds the L×N Toeplitz matrix using:
 *   c = sv[L-1::-1],  r = sv[L-1:]
 * and
 *   A(i,j) = (i >= j) ? c[i-j] : r[j-i]
 *
 * Requirements:
 *   - L > 0, N > 0, and L < N
 *   - starting_vector length == L + N - 1
 *
 */

#ifndef __toeplitz_standard_h
#define __toeplitz_standard_h

#include "config.h"
#include "matrix.h"
#include "vector.h"
#include <cassert>

namespace libbase
{

class toeplitz_standard
{
public:
    // Single template parameter T (bool, int, double, ...).
    template <class T>
    static matrix<T> build(const vector<T>& sv, int L, int N)
    {
        // Strict preconditions
        assertalways(L > 0 && N > 0);
        assertalways(L < N);

        // starting_vector must be exactly L + N - 1
        const int sv_len = L + N - 1;
        assertalways(sv.size().length() == sv_len);

        matrix<T> A(L, N);
        const int Lm1 = L - 1;

        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < N; ++j) {
                // SciPy-equivalent indexing:
                // if j <= i: A(i,j) = sv[(L-1) - (i - j)]
                // else     : A(i,j) = sv[(L-1) + (j - i)]
                const int sv_idx = (j <= i) ? (Lm1 - (i - j)) : (Lm1 + (j - i));
                assert(sv_idx >= 0 && sv_idx < sv.size().length());
                A(i, j) = sv(sv_idx);
            }
        }
        return A;
    }
};

} // namespace libbase

#endif // __toeplitz_standard_h
