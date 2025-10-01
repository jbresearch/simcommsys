/*!
 * \file Base class for Privacy Amplification to get the final secret (hashed)
 * key.
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
 * \file Base class for Privacy Amplification to get the final secret (hashed)
 * key.
 */

#ifndef __privacy_amplification_h
#define __privacy_amplification_h

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits> // for std::is_same, std::is_integral

#include "assertalways.h"
#include "matrix.h"
#include "randgen.h"
#include "random.h"
#include "field_utils.h"
#include "serializer.h"
#include "vector.h"

namespace libcomm
{

template <class T>
class privacy_amplification_base : public libbase::serializable
{
protected:
    libbase::randgen rng;
    libbase::vector<T> hashed_key;

public:
    virtual void seedfrom(libbase::random& r) = 0;

    // Generates a random starting vector of the requested length.
    libbase::vector<T> generate_starting_vector(int starting_vector_len,
                                                int alphabet_size)
    {
        assert(alphabet_size >= 2);
        assert(starting_vector_len > 0);

        libbase::vector<T> starting_vector(starting_vector_len);

        if (std::is_same<T, bool>::value) {
            assert(alphabet_size == 2);
            for (int i = 0; i < starting_vector_len; ++i) {
                starting_vector(i) = (rng.ival(2) != 0);
            }
        } else if (std::is_integral<T>::value) {
            for (int i = 0; i < starting_vector_len; ++i)
                starting_vector(i) = static_cast<T>(rng.ival(alphabet_size));
        } else { // Case of GF field.
            for (int i = 0; i < starting_vector_len; ++i)
                starting_vector(i) = static_cast<T>(rng.ival(2));
        }

        return starting_vector;
    }

    // To be implemented by subclass.
    virtual int generate_starting_vector_length() = 0;

    // To be implemented by subclass: must return an L×N Toeplitz matrix.
    virtual libbase::matrix<T>
    generate_toeplitz_matrix(const libbase::vector<T>& starting_vector) = 0;

    // Compute y = Toeplitz * key using matrix.h.
    // Uses (Toeplitz.transpose()) * key to avoid building an N×1 matrix.
    // Handles both bool, int and GF types.
    libbase::vector<T>
    compute_hashed_key(const libbase::matrix<T>& toeplitz_matrix,
                       const libbase::vector<T>& pre_hashed_key)
    {
        // ---- sanity checks ----
        assert(toeplitz_matrix.size().rows() > 0);
        assert(toeplitz_matrix.size().cols() > 0);
        assert(pre_hashed_key.size() > 0);

        // ---- Case using bool.
        if (std::is_same<T, bool>::value) {
            const int L = toeplitz_matrix.size().rows();
            const int N = toeplitz_matrix.size().cols();

            libbase::matrix<int> A_int(toeplitz_matrix); // L×N
            libbase::vector<int> x_int(N);
            for (int j = 0; j < N; ++j)
                x_int(j) = pre_hashed_key(j) ? 1 : 0;

            // matrix.h operator*(vector) computes A^T * x, so use A^T
            libbase::vector<int> y_int =
                (A_int.transpose()) * x_int; // length L

            libbase::vector<T> y(L);
            for (int i = 0; i < L; ++i)
                y(i) = (y_int(i) & 1) != 0;

            hashed_key = y;
            assert(hashed_key.size() == L);
            return hashed_key;
        }

        // ---- Case using int.
        if (std::is_integral<T>::value) {
            const int alphabet_size = field_utils<T>::elements();
            assert(alphabet_size >= 2);

            // y = (A^T) * x
            libbase::vector<T> y =
                (toeplitz_matrix.transpose()) * pre_hashed_key; // length L

            // Reduce modulo alphabet_size element-wise
            for (int i = 0; i < y.size(); ++i) {
                int r = static_cast<int>(y(i)) % alphabet_size;
                if (r < 0)
                    r += alphabet_size;
                y(i) = static_cast<T>(r);
            }

            hashed_key = y;
            assert(hashed_key.size() == toeplitz_matrix.size().rows());
            return hashed_key;
        }

        // ---- Case: using GF fields.
        {
            libbase::vector<T> y =
                (toeplitz_matrix.transpose()) * pre_hashed_key; // length L
            hashed_key = y;
            assertalways(hashed_key.size() == toeplitz_matrix.size().rows());
            return hashed_key;
        }
    }

    // Helper getter fn to get hashed_key.
    const libbase::vector<T>& get_hashed_key() const { return hashed_key; }

    virtual ~privacy_amplification_base() {}

    //! \brief Description of serializable object
    virtual std::string description() const = 0;

    using privacy_amplification = privacy_amplification_base<T>;

    // Serialization Support
    DECLARE_BASE_SERIALIZER(privacy_amplification) // templated case
};

} // namespace libcomm

#endif // __privacy_amplification_h
