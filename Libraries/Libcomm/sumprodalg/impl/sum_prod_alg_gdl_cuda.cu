/*!
 * \file
 *
 * Copyright (c) 2024 Mark Mizzi
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

#include "cuda/cuda_assert.h"
#include "cuda/matrix.h"
#include "cuda/vector.h"
#include "gf.h"
#include "sum_prod_alg_gdl_cuda.h"
#include <cmath>
#include <limits>

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

/*! \brief struct wrapper for std::numeric_limits<real>::epsilon().
 *
 * This is needed because constexpr functions are not expanded when compiling
 * device code. This leads to an error where the compiler thinks we are trying
 * to call a host function.
 *
 * The use of static constexpr fields on structs like this is allowed, as it is
 * expanded by the compiler.
 */
template <class real>
struct epsilon {
    static constexpr real val = std::numeric_limits<real>::epsilon();
};

/*! \brief Perform clipping of zero values to almost-zero values on device.
 */
template <class real>
__device__
void
perform_clipping(real& num, int& clipping_method, real& almost_zero)
{
    if (1 == clipping_method) {
        // use standard clipping
        num = max(num, almost_zero);
    } else {
        // branchless computation.
        num = (num <= real(0.0)) * almost_zero + (num > real(0.0)) * num;
    }
}

template <class GF_q, class real>
__global__ void
normalize_probs_kern(::cuda::matrix_reference<real> recvd_probs,
                     int clipping_method,
                     real almost_zero)
{
    int loop_n = blockIdx.x * blockDim.x + threadIdx.x;

    // bounds checking
    int dim_n = recvd_probs.get_rows();
    loop_n = min(loop_n, dim_n - 1);

    int num_of_elements = GF_q::elements();
    real tmp_prob;
    real alpha = real(0.0);

    // perform clipping of zero values to almost zero
    // also accumulate the sum of all probabilities into alpha
    for (int loop_e = 0; loop_e < num_of_elements; loop_e++) {
        // Clipping HACK
        tmp_prob = recvd_probs(loop_n, loop_e);
        perform_clipping(tmp_prob, clipping_method, almost_zero);
        recvd_probs(loop_n, loop_e) = tmp_prob;
        alpha += tmp_prob;
    }
    cuda_assertalways(alpha != real(0.0));

    // normalize probabilities (divide by alpha)
    for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
        recvd_probs(loop_n, loop_e) /= alpha;
}

/*! \brief compute the Fast Hadamard transform
 * This method will compute the Fast Fourier Transform of the
 * elements passed in through conv_out. It does this recursively.
 * Note the result is equivalent to the following matrix-vector
 * multiplication:
 * Let m be the size of conv_out, ie m=|GF_q|=power of 2
 * Let H_m be the standard (mxm)-Hadamard matrix, ie
 * H_2k=H_2 "*" H_k where "*" is the Kronecker product of 2 matrices and
 *      [ 1   1 ]
 * H_2= [       ]
 *      [ 1  -1 ]
 * then the result of this method is equal to H_m*conv_out^t where
 * conv_out^t is the transpose of the conv_out vector
 *
 */
template <class real>
__device__
void
compute_convs(::cuda::vector_reference<real> conv_out, int pos1, int pos2)
{
    // this is in fact the Hadamard transform using the butterfly property
    // of the fast Fourier transform.
    if ((pos2 - pos1) == 1) {
        real tmp1 = conv_out(pos1);
        real tmp2 = conv_out(pos2);
        conv_out(pos1) = tmp1 + tmp2;
        conv_out(pos2) = tmp1 - tmp2;
    } else {
        int midpoint = pos1 + (pos2 - pos1 + 1) / 2;
        // NOTE: Compute H_{m-1}x_u, and store in first half of conv_out
        // Here H_{m-1} is a Hadamard matrix one less than the current one,
        // and x_u is upper half of conv_out
        compute_convs(conv_out, pos1, midpoint - 1);
        // NOTE: Compute H_{m-1}x_b, and store in second half of conv_out
        // Here H_{m-1} is a Hadamard matrix one less than the current one,
        // and x_b is lower half of conv_out
        compute_convs(conv_out, midpoint, pos2);
        pos2 = midpoint;
        // NOTE: Iterate over H_{m-1}x_u and H_{m-1}x_b at the same time.
        // Both are vectors of length 2^{m-1} stored in upper and lower half
        // of conv_out.
        for (int loop1 = pos1; loop1 < midpoint; loop1++) {
            // NOTE: Get (H_{m-1}x_u)[i]
            real tmp1 = conv_out(loop1);
            // NOTE: Get (H_{m-1}x_b)[i]
            real tmp2 = conv_out(pos2);
            // NOTE: Here we are effectively computing ith and i + 2^{m-1}th
            // elements of the transform:

            // [H_{m-1}     H_{m-1}] [x_u] = [H_{m-1}x_u + H_{m-1}x_b]
            //  [H_{m-1}    -H_{m-1}] [x_b]   [H_{m-1}x_u - H_{m-1}x_b]

            // NOTE: Compute (H_{m-1}x_u)[i] + (H_{m-1}x_b)[i]
            conv_out(loop1) = tmp1 + tmp2;
            // NOTE: Compute (H_{m-1}x_u)[i] - (H_{m-1}x_b)[i]
            conv_out(pos2) = tmp1 - tmp2;
            pos2++;
        }
    }
}

template <class GF_q, class real>
__global__ void
spa_init_kern(::cuda::matrix_reference<int> device_perms,
              ::cuda::matrix_reference<real> device_received_probs,
              ::cuda::matrix_reference<int> device_qmn_row_indices,
              ::cuda::matrix_reference<real> device_r_mxn,
              ::cuda::matrix_reference<real> device_q_mxn,
              ::cuda::matrix_reference<real> device_qmn_conv,
              ::cuda::vector_reference<int> device_pchk_row_non_zeros,
              ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
              ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val)
{
    // find loop_m
    int loop_m = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int m = device_pchk_row_non_zeros.size();
    loop_m = min(loop_m, m - 1);

    // find loop_e
    int loop_e = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements);

    int non_zeros = device_pchk_row_non_zeros(loop_m);

    int qmn_row_idx;
    int pos;
    GF_q h_m_n;
    // NOTE: loop_n iterates over number of bits that participate in mth
    // check of a codeword. E.g. if check involves {x_1, x_4, x_6}, loop_n
    // ranges over [0, 1, 2]
    for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
        // NOTE: pos is the actual index of the nth bit participating in the
        // mth check in the codeword. E.g. if check involves {x_1, x_4, x_6}
        // and loop_n = 1, pos = 4 (-1 since we count from 0)
        pos = device_pchk_row_non_zeros_pos(loop_m, loop_n);
        // NOTE: Find corresponding value in the parity check matrix.
        // We use loop_m because this is the check index, and pos because
        // this is the actual index of the nth bit participating in the mth
        // check (non_zeros variable does not count bits that don't
        // participate in the mth check).
        h_m_n = device_pchk_row_non_zeros_val(loop_m, loop_n);

        // NOTE: Initially we set (probability distribution) q_mn
        // (which is prob. of bit n having value x given info. of all checks
        // other than m) to simply the prior probability distribution of bit
        // n
        qmn_row_idx = device_qmn_row_indices(loop_m, pos);
        device_q_mxn(qmn_row_idx, loop_e) = device_received_probs(pos, loop_e);

        // In fact the probability we are given are not for the x_i but
        // for the value h_m_n*xi hence all we need to do is copy the
        // values into the array with a slightly amended index:
        // probs(h_m_n*x)=received_prob(x) for all x in GF_q and
        // 0!=h_m_n in GF_q.
        // Declerq&Fossorier: Decoding Algs for non-binary LDPC Codes
        // over GF(q)
        // perms(h_m_n)(loop)=GF_q(h_m_n)*GF_q(loop) - a look-up is
        // quicker than a computation (I hope)

        // NOTE: Here we are permuting the prior probability
        // distribution by multiplying it with h_m_n before placing
        // it in the qmn_conv array.
        device_qmn_conv(qmn_row_idx, device_perms(h_m_n, loop_e)) =
            device_received_probs(pos, loop_e);

        compute_convs(
            device_qmn_conv.extract_row(qmn_row_idx), 0, num_of_elements - 1);

        // r_mxn is initialized as 0.
        device_r_mxn(qmn_row_idx, loop_e) = 0.0;
    }
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::spa_init(const array1vd_t& recvd_probs)
{
    dim3 block_dim;
    dim3 num_blocks;

    int num_of_elements = GF_q::elements();
    int dim_n = recvd_probs.size();

    // Allocate memory for recieved probabilities.
    this->device_received_probs.init(dim_n, num_of_elements);

    // Copy probabilities to device row-by-row since they are provided in a
    // ragged list.
    for (int loop_n = 0; loop_n < dim_n; loop_n++)
        this->device_received_probs.extract_row(loop_n) = recvd_probs(loop_n);

    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(-(-dim_n / block_dim.x));
    // normalize probabilities (and also convert zeros to almost zeros)
    normalize_probs_kern<GF_q, real><<<block_dim, num_blocks>>>(
        ::cuda::matrix_reference<real>(device_received_probs),
        this->clipping_method,
        this->almostzero);

    // TODO: Fix this.
#if DEBUG >= 2
    libbase::trace << std::endl
                   << "The first 5 normalised likelihoods are given by:"
                   << std::endl;
    libbase::trace << this->received_probs.extract(0, 5);
#endif

    // ----------------------------------
    // Continue

    // this uses the description of the algorithm as given by

    // MacKay in Information Theory, Inference and Learning Algorithms(2003)
    // on page 560 - chapter 47.3

    // some helper variables

    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks =
        dim3(-(-dim_n / block_dim.x), -(-num_of_elements / block_dim.y));
    spa_init_kern<GF_q, real>
        <<<block_dim, num_blocks>>>(this->device_perms,
                                    this->device_received_probs,
                                    this->device_qmn_row_indices,
                                    this->device_r_mxn,
                                    this->device_q_mxn,
                                    this->device_qmn_conv,
                                    this->device_pchk_row_non_zeros,
                                    this->device_pchk_row_non_zeros_pos,
                                    this->device_pchk_row_non_zeros_val);

    // TODO: Fix this.
#if DEBUG >= 2
    libbase::trace << " Memory Usage:\n ";
    libbase::trace << this->marginal_probs.size() *
                          sizeof(sum_prod_alg_abstract<GF_q, real>::marginals) /
                          double(1 << 20)
                   << " MB" << std::endl;

    libbase::trace << std::endl
                   << "The marginal matrix is given by:" << std::endl;
    this->print_marginal_probs(libbase::trace);
#endif
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::spa_iteration(array1vd_t& ro)
{
}

} // namespace libcomm

#include "logrealfast.h"
#include "mpreal.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

using libbase::logrealfast;
using libbase::mpreal;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define REAL_TYPE_SEQ \
      (double)(logrealfast)(mpreal)

/* Serialization string: ldpc<type,real>
 * where:
 *      type = gf2 | gf4 ...
 *      real = double | logrealfast | mpreal
 */
#define INSTANTIATE(r, args) \
      template class sum_prod_alg_gdl_cuda<BOOST_PP_SEQ_ENUM(args)>;
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // namespace libcomm
