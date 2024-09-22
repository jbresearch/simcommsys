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
#include "cuda/stream.h"
#include "cuda/util.h"
#include "cuda/vector.h"
#include "gf.h"
#include "hard_decision.h"
#include "sum_prod_alg_gdl_cuda.h"
#include "vector.h"
#include <cmath>
#include <limits>
#include <memory>

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

template <class GF_q, class real>
__global__ void
seed_hd_functor(
    basic_hard_decision<real, GF_q, ::cuda::vector_reference<real>>* hd_functor,
    libbase::int32u rval)
{
    hd_functor->seed(rval);
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::seedfrom(libbase::random& r)
{
    // Call base method first
    Base::seedfrom(r);

    int device = ::cuda::cudaGetCurrentDevice();
    seed_hd_functor<<<1, 1>>>(this->hd_functor.get(), r.ival());
}

/*! \brief Compute ceil(X / Y)
 */
#define ROUND_UP_DIV(X, Y) (((X) + (Y) - 1) / (Y))

template <class GF_q, class real>
__global__ void
hadamard_transform_pass_kern(::cuda::matrix_reference<real, false> src,
                             ::cuda::matrix_reference<real, false> dst,
                             int tanner_edges,
                             int h)
{
    int num_of_elements = GF_q::elements();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int loop_e = idx % num_of_elements;

    // this is just a generic index that ranges over [0, tanner_edges)
    int i = idx / num_of_elements;
    i = min(i, tanner_edges - 1);

    // If floor(loop_e / h) is odd, sign is -1.0
    // If floor(loop_e / h) is even, sign is 1.0
    int sign = ((real)((loop_e / h) % 2 == 0) - 0.5) * 2.0;

    // From the butterfly property:
    // If floor(loop_e / h) is odd, result of the pass is P[loop_e - h] - P[e]
    // If floor(loop_e / h) is even, result of the pass is P[loop_e + h] + P[e]
    dst(i, loop_e) = src(i, loop_e + sign * h) + sign * src(i, loop_e);
}

template <class GF_q, class real>
__global__ void
multiply_h_m_n_kern(
    ::cuda::matrix_reference<int, false> device_qmn_row_nxm_indices,
    ::cuda::matrix_reference<real, false> src,
    ::cuda::matrix_reference<real, false> dst,
    ::cuda::vector_reference<int> device_pchk_col_non_zeros,
    ::cuda::matrix_reference<GF_q, false> device_pchk_col_non_zeros_val)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    int num_of_elements = GF_q::elements();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // find loop_e
    int loop_e = idx % num_of_elements;

    // find pos_n
    int pos_n = idx / num_of_elements;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    pos_n = min(pos_n, n - 1);

    int non_zeros = device_pchk_col_non_zeros(pos_n);
    // hold value of pchk matrix at (m, n)
    GF_q h_m_n;
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        h_m_n = device_pchk_col_non_zeros_val(pos_n, loop_m);
        // perform the permutation
        dst(device_qmn_row_nxm_indices(pos_n, loop_m), h_m_n * GF_q(loop_e)) =
            src(device_qmn_row_nxm_indices(pos_n, loop_m), loop_e);
    }
}

template <class GF_q, class real>
__global__ void
divide_h_m_n_kern(
    ::cuda::matrix_reference<int, false> device_qmn_row_nxm_indices,
    ::cuda::matrix_reference<real, false> src,
    ::cuda::matrix_reference<real, false> dst,
    ::cuda::vector_reference<int> device_pchk_col_non_zeros,
    ::cuda::matrix_reference<GF_q, false> device_pchk_col_non_zeros_val)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    int num_of_elements = GF_q::elements();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // find loop_e
    int loop_e = idx % num_of_elements;

    // find pos_n
    int pos_n = idx / num_of_elements;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    pos_n = min(pos_n, n - 1);

    int non_zeros = device_pchk_col_non_zeros(pos_n);
    // hold value of pchk matrix at (m, n)
    GF_q h_m_n;
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        h_m_n = device_pchk_col_non_zeros_val(pos_n, loop_m);
        // perform the permutation
        dst(device_qmn_row_nxm_indices(pos_n, loop_m), loop_e) = src(
            device_qmn_row_nxm_indices(pos_n, loop_m), h_m_n * GF_q(loop_e));
    }
}

template <class GF_q, class real>
inline void
hadamard_transform(::cuda::matrix_reference<real, false>& src,
                   ::cuda::matrix_reference<real, false>& dst)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    int num_of_elements = GF_q::elements();
    int tanner_edges = src.get_rows();

    dim3 block_dim(1024);
    // use division which truncates upwards.
    dim3 num_blocks(
        ROUND_UP_DIV(num_of_elements * tanner_edges, (int)block_dim.x));

    int h;
    for (h = 1; h < num_of_elements; h <<= 1) {
        hadamard_transform_pass_kern<GF_q, real>
            <<<num_blocks, block_dim>>>(src, dst, tanner_edges, h);
        cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
        cudaDeviceSynchronize();
#endif

        std::swap(src, dst);
    }
}

template <class GF_q, class real>
sum_prod_alg_gdl_cuda<GF_q, real>::sum_prod_alg_gdl_cuda(
    int n,
    int m,
    const array1vi_t& non_zero_col_pos,
    const array1vi_t& non_zero_row_pos,
    const libbase::matrix<GF_q>& pchk_matrix)
{
    int num_of_elements = GF_q::elements();

    // We also build the various parity check matrix fields on the host,
    // then copy to the device.
    array1i_t pchk_row_non_zeros(m);
    array1i_t pchk_col_non_zeros(n);

    // Find the maximum number of non zero elements in a row of the parity
    // check matrix.
    // Also populate pchk_row_non_zeros.
    int non_zeros = 0;
    max_pchk_row_non_zeros = std::numeric_limits<int>::min();
    for (int loop_m = 0; loop_m < m; loop_m++) {
        non_zeros = non_zero_row_pos(loop_m).size();

        pchk_row_non_zeros(loop_m) = non_zeros;
        max_pchk_row_non_zeros = std::max(max_pchk_row_non_zeros, non_zeros);
    }

    matrixi_t pchk_row_non_zeros_pos(m, max_pchk_row_non_zeros);
    libbase::matrix<GF_q> pchk_row_non_zeros_val(m, max_pchk_row_non_zeros);

    // Populate per-row representation of the parity check matrix.
    int pos_n;
    for (int pos_m = 0; pos_m < m; pos_m++) {
        // non-zeros for this row of the parity check matrix
        non_zeros = pchk_row_non_zeros(pos_m);

        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            pos_n = non_zero_row_pos(pos_m)(loop_n) - 1; // we count from zero;

            // populate other pchk matrix fields on the host.
            pchk_row_non_zeros_pos(pos_m, loop_n) = pos_n;
            pchk_row_non_zeros_val(pos_m, loop_n) = pchk_matrix(pos_m, pos_n);
        }
    }

    // Find the maximum number of non zero elements in a col of the parity
    // check matrix.
    // Also populate pchk_col_non_zeros.
    max_pchk_col_non_zeros = std::numeric_limits<int>::min();
    for (int loop_n = 0; loop_n < n; loop_n++) {
        non_zeros = non_zero_col_pos(loop_n).size();

        pchk_col_non_zeros(loop_n) = non_zeros;
        max_pchk_col_non_zeros = std::max(max_pchk_col_non_zeros, non_zeros);
    }

    libbase::matrix<GF_q> pchk_col_non_zeros_val(n, max_pchk_col_non_zeros);

    // we first build qmn_row_nxm_indices, qmn_row_mxn_indices on the host,
    // then copy to device. Easier since this operation is inherently serial (we
    // have a counter to keep track of current index) and also we need the
    // tanner_edges var computed during this process on host to allocate
    // memory for qmn and rmn matrices.
    matrixi_t qmn_row_nxm_indices(n, max_pchk_col_non_zeros);
    matrixi_t qmn_row_mxn_indices(m, max_pchk_row_non_zeros);

    // counts the number of edges in the Tanner graph of the code.
    // Tells us what the size of device_rmxn and device_qmn_conv should
    // be.
    int tanner_edges = 0;

    // Populate qmn_row_nxm_indices, qmn_row_mxn_indices
    // Also populate the rest of the parity check matrix repr. on the host.
    // Actual m value, since loop_m is just an index ranging over the number
    // of non-zero values in a col of pchk_matrix.
    int pos_m;
    for (int pos_n = 0; pos_n < n; pos_n++) {
        // non-zeros for this col of the parity check matrix
        non_zeros = pchk_col_non_zeros(pos_n);

        for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
            pos_m = non_zero_col_pos(pos_n)(loop_m) - 1; // we count from zero;

            // populate other pchk matrix fields on the host.
            pchk_col_non_zeros_val(pos_n, loop_m) = pchk_matrix(pos_m, pos_n);

            // linear search for loop_n; should be fast as pchk matrix is
            // sparse.
            int loop_n = -1;
            for (int loop_n_dash = 0; loop_n_dash < pchk_row_non_zeros(pos_m);
                 loop_n_dash++)
                if (non_zero_row_pos(pos_m)(loop_n_dash) - 1 == pos_n) {
                    loop_n = loop_n_dash;
                    break;
                }
            assert(loop_n >= 0);

            // assign an index in device_q_mn_conv, device_r_mxn and so on
            // to a non-zero (m, n) element.
            qmn_row_nxm_indices(pos_n, loop_m) = tanner_edges;
            qmn_row_mxn_indices(pos_m, loop_n) = tanner_edges;
            tanner_edges++;
        }
    }

    device_qmn_row_nxm_indices.init(n, max_pchk_col_non_zeros);
    device_qmn_row_mxn_indices.init(m, max_pchk_row_non_zeros);
    // Copy qmn_row_nxm_indices, qmn_row_nxm_indices to device
    device_qmn_row_nxm_indices = qmn_row_nxm_indices;
    device_qmn_row_mxn_indices = qmn_row_mxn_indices;

    // Allocate memory on the device for representation of the parity check
    // matrix
    device_pchk_row_non_zeros.init(m);
    device_pchk_row_non_zeros_pos.init(m, max_pchk_row_non_zeros);
    device_pchk_row_non_zeros_val.init(m, max_pchk_row_non_zeros);

    device_pchk_col_non_zeros.init(n);
    device_pchk_col_non_zeros_val.init(n, max_pchk_col_non_zeros);

    // Copy represenation of the parity check matrix to the device.
    device_pchk_row_non_zeros = pchk_row_non_zeros;
    device_pchk_row_non_zeros_pos = pchk_row_non_zeros_pos;
    device_pchk_row_non_zeros_val = pchk_row_non_zeros_val;

    device_pchk_col_non_zeros = pchk_col_non_zeros;
    device_pchk_col_non_zeros_val = pchk_col_non_zeros_val;

    // Allocate required memory for r_mxn, q_mxn and qmn_conv on device.
    device_r_mxn.init(tanner_edges, num_of_elements);
    device_qmn_conv.init(tanner_edges, num_of_elements);

    device_swap_buf.init(tanner_edges, num_of_elements);

    device_out_probs.init(n, num_of_elements);

    device_received_word.init(n);
    device_syndrome.init(m);
}

template <class real>
__device__
void
perform_clipping(real& num, int& clipping_method, real& almostzero)
{
    if (1 == clipping_method) {
        // use standard clipping
        num = max(num, almostzero);
    } else {
        // branchless computation.
        num = (num <= real(0.0)) * almostzero + (num > real(0.0)) * num;
    }
}

template <class GF_q, class real>
__global__ void
clip_and_normalize_probs_kern(::cuda::matrix_reference<real, false> probs,
                              int clipping_method,
                              real almostzero)
{
    // ranges over probability distributions in prob.
    int loop_n = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int n = probs.get_rows();
    loop_n = min(loop_n, n - 1);

    int num_of_elements = GF_q::elements();
    real alpha = real(0.0);

    real tmp_prob;
    for (int loop_e = 0; loop_e < num_of_elements; loop_e++) {
        // Clipping HACK
        tmp_prob = probs(loop_n, loop_e);
        perform_clipping(tmp_prob, clipping_method, almostzero);
        probs(loop_n, loop_e) = tmp_prob;
        alpha += tmp_prob;
    }

    cuda_assertalways(alpha != real(0.0));

    // normalize probabilities (divide by alpha)
    for (int loop_e = 0; loop_e < num_of_elements; loop_e++) {
        probs(loop_n, loop_e) /= alpha;
    }
}

template <class GF_q, class real>
inline void
clip_and_normalize_probs(::cuda::matrix_reference<real, false> probs,
                         int clipping_method,
                         real almostzero)
{
    int n = probs.get_rows();

    dim3 block_dim(1024);
    dim3 num_blocks(ROUND_UP_DIV(n, (int)block_dim.x));

    clip_and_normalize_probs_kern<GF_q, real>
        <<<num_blocks, block_dim>>>(probs, clipping_method, almostzero);
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
}

template <class GF_q, class real>
__global__ void
spa_init_kern(
    ::cuda::matrix_reference<real, false> device_received_probs,
    ::cuda::matrix_reference<int, false> device_qmn_row_nxm_indices,
    ::cuda::matrix_reference<real, false> device_r_mxn,
    ::cuda::matrix_reference<real, false> device_qmn_conv,
    ::cuda::vector_reference<int> device_pchk_col_non_zeros,
    ::cuda::matrix_reference<GF_q, false> device_pchk_col_non_zeros_val)
{
    int num_of_elements = GF_q::elements();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // find loop_e
    int loop_e = idx % num_of_elements;

    // find pos_n
    int pos_n = idx / num_of_elements;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    pos_n = min(pos_n, n - 1);

    int non_zeros = device_pchk_col_non_zeros(pos_n);

    int qmn_row_idx;
    GF_q h_m_n;
    // NOTE: loop_m iterates over the number of checks that symbol pos_n
    // participates in.
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        // NOTE: Find corresponding value in the parity check matrix.
        h_m_n = device_pchk_col_non_zeros_val(pos_n, loop_m);

        // get index into device_qmn_conv and device_r_mxn
        qmn_row_idx = device_qmn_row_nxm_indices(pos_n, loop_m);

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
        device_qmn_conv(qmn_row_idx, h_m_n * GF_q(loop_e)) =
            device_received_probs(pos_n, loop_e);
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

    for (int loop_n = 0; loop_n < dim_n; loop_n++)
        this->device_received_probs.extract_row(loop_n) = recvd_probs(loop_n);

    clip_and_normalize_probs<GF_q, real>(
        this->device_received_probs, this->clipping_method, this->almostzero);

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

    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * dim_n, (int)block_dim.x));
    spa_init_kern<GF_q, real>
        <<<num_blocks, block_dim>>>(this->device_received_probs,
                                    this->device_qmn_row_nxm_indices,
                                    this->device_r_mxn,
                                    this->device_qmn_conv,
                                    this->device_pchk_col_non_zeros,
                                    this->device_pchk_col_non_zeros_val);
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // apply the FFT again to get the proper values
    // Here we use matrix references for cheap swapping. The result of the
    // Hadamard transform will always be in src.
    ::cuda::matrix_reference<real, false> src(device_qmn_conv);
    ::cuda::matrix_reference<real, false> dst(device_swap_buf);
    hadamard_transform<GF_q, real>(src, dst);

    // Result of the Hadamard transform is always stored in first arg passed to
    // hadamard_transform(), copy to device_qmn_conv in case src is the swap
    // buffer.
    device_qmn_conv = src;

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
__global__ void
compute_r_mn_kern(
    ::cuda::matrix_reference<int, false> device_qmn_row_mxn_indices,
    ::cuda::matrix_reference<real, false> device_r_mxn,
    ::cuda::matrix_reference<real, false> device_qmn_conv,
    ::cuda::vector_reference<int> device_pchk_row_non_zeros)
{
    int num_of_elements = GF_q::elements();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // find loop_e
    int loop_e = idx % num_of_elements;

    // find loop_m
    int pos_m = idx / num_of_elements;
    // bounds checking
    int m = device_pchk_row_non_zeros.size();
    pos_m = min(pos_m, m - 1);

    int non_zeros = device_pchk_row_non_zeros(pos_m);
    // Holds the actual message computed
    real q_nm_conv_prod;
    for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
        q_nm_conv_prod = 1.0;
        for (int loop_n_dash = 0; loop_n_dash < non_zeros; loop_n_dash++) {
            // NOTE: Branchless computation
            q_nm_conv_prod *=
                // Branch where pos_n_dash != pos_n and we
                // include the corresponding qmn in the
                // message
                (loop_n_dash != loop_n) *
                    device_qmn_conv(
                        device_qmn_row_mxn_indices(pos_m, loop_n_dash),
                        loop_e) +
                // Branch where we multiply by 1, effectively removing
                // q_nm for pos_n from the computed message.
                (loop_n_dash == loop_n);
        }
        // Loop above has potential divergence as different m have different
        // degrees in general. We want to convergence again here so most iters
        // are in sync.
        // TODO: Test impact of this.
        __syncthreads();
        // coalesced memory access due to syncthreads above
        // We store in r_mxn but this is not the final result.
        device_r_mxn(device_qmn_row_mxn_indices(pos_m, loop_n), loop_e) =
            q_nm_conv_prod;
    }
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::compute_r_mn()
{
    dim3 block_dim, num_blocks;

    int m = device_pchk_row_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * m, (int)block_dim.x));
    compute_r_mn_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int, false>(device_qmn_row_mxn_indices),
        ::cuda::matrix_reference<real, false>(device_r_mxn),
        ::cuda::matrix_reference<real, false>(device_qmn_conv),
        ::cuda::vector_reference<int>(device_pchk_row_non_zeros));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // apply the FFT again to get the proper values
    // Here we use matrix references for cheap swapping.
    ::cuda::matrix_reference<real, false> src(device_r_mxn);
    ::cuda::matrix_reference<real, false> dst(device_swap_buf);
    hadamard_transform<GF_q, real>(src, dst);

    int n = device_pchk_col_non_zeros.size();
    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * n, (int)block_dim.x));

    // Permute the distributions in src (transformed by the Hadamard transform)
    // into dst
    divide_h_m_n_kern<<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int, false>(device_qmn_row_nxm_indices),
        src,
        dst,
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros),
        ::cuda::matrix_reference<GF_q, false>(device_pchk_col_non_zeros_val));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // dst could be device_r_mxn or device_swap_buf depending on whether no. of
    // passes in Hadamard transform is even or odd. We copy back to device_r_mxn
    // to make sure the result is in the right array.
    device_r_mxn = dst;

    // Apply clipping + normalization to the computed r_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real, false>(device_r_mxn),
        this->clipping_method,
        this->almostzero);
}

template <class GF_q, class real>
__global__ void
compute_q_mn_kern(
    ::cuda::matrix_reference<real, false> device_received_probs,
    ::cuda::matrix_reference<int, false> device_qmn_row_nxm_indices,
    ::cuda::matrix_reference<real, false> device_r_mxn,
    ::cuda::matrix_reference<real, false> device_qmn_conv,
    ::cuda::vector_reference<int> device_pchk_col_non_zeros)
{
    int num_of_elements = GF_q::elements();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // find loop_e
    int loop_e = idx % num_of_elements;

    // find loop_n
    int pos_n = idx / num_of_elements;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    pos_n = min(pos_n, n - 1);

    // Current probability that received symbol n has value e.
    real recvd_prob = device_received_probs(pos_n, loop_e);

    int non_zeros = device_pchk_col_non_zeros(pos_n);
    // Holds the actual message computed
    real q_nm;
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        q_nm = recvd_prob;

        for (int loop_m_dash = 0; loop_m_dash < non_zeros; loop_m_dash++) {
            // NOTE: Branchless computation
            q_nm *=
                // Branch where pos_m_dash != pos_m and we
                // include the corresponding r_mn in the
                // message
                (loop_m_dash != loop_m) *
                    device_r_mxn(device_qmn_row_nxm_indices(pos_n, loop_m_dash),
                                 loop_e) +
                // Branch where we multiply by 1, effectively removing
                // r_mn for pos_m from the computed message.
                (loop_m_dash == loop_m);
        }
        // Loop above has potential divergence as different m have different
        // degrees in general. We want to convergence again here so most iters
        // are in sync.
        // TODO: Test impact of this.
        __syncthreads();

        // Uncoalesced memory access.
        device_qmn_conv(device_qmn_row_nxm_indices(pos_n, loop_m), loop_e) =
            q_nm;
        // resynchronize after uncoalesced memory access
        __syncthreads();
    }
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::compute_q_mn()
{

    dim3 block_dim, num_blocks;

    int n = device_pchk_col_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * n, (int)block_dim.x));
    compute_q_mn_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<real, false>(device_received_probs),
        ::cuda::matrix_reference<int, false>(device_qmn_row_nxm_indices),
        ::cuda::matrix_reference<real, false>(device_r_mxn),
        ::cuda::matrix_reference<real, false>(device_qmn_conv),
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // Apply clipping + normalization to the computed q_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real, false>(device_qmn_conv),
        this->clipping_method,
        this->almostzero);

    // Here we use matrix references for cheap swapping.
    ::cuda::matrix_reference<real, false> src(device_qmn_conv);
    ::cuda::matrix_reference<real, false> dst(device_swap_buf);

    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * n, (int)block_dim.x));

    // Permute the distributions in src into dst
    multiply_h_m_n_kern<<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int, false>(device_qmn_row_nxm_indices),
        src,
        dst,
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros),
        ::cuda::matrix_reference<GF_q, false>(device_pchk_col_non_zeros_val));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // Compute Hadamard transform on the result.
    hadamard_transform<GF_q, real>(dst, src);

    // Result of the Hadamard transform is always stored in first arg passed to
    // hadamard_transform(), copy to device_qmn_conv in case dst is the swap
    // buffer.
    device_qmn_conv = dst;
}

template <class GF_q, class real>
__global__ void
compute_probs_kern(
    ::cuda::matrix_reference<real, false> device_received_probs,
    ::cuda::matrix_reference<real, false> device_out_probs,
    ::cuda::matrix_reference<int, false> device_qmn_row_nxm_indices,
    ::cuda::matrix_reference<real, false> device_r_mxn,
    ::cuda::vector_reference<int> device_pchk_col_non_zeros)
{
    int num_of_elements = GF_q::elements();
    // find loop_e
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int loop_e = idx % num_of_elements;

    // find loop_n
    int pos_n = idx / num_of_elements;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    pos_n = min(pos_n, n - 1);

    int non_zeros = device_pchk_col_non_zeros(pos_n);
    // Holds the prob computed
    real prob = device_received_probs(pos_n, loop_e);
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        prob *= device_r_mxn(device_qmn_row_nxm_indices(pos_n, loop_m), loop_e);
    }

    device_out_probs(pos_n, loop_e) = prob;
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::compute_probs()
{
    dim3 block_dim, num_blocks;

    int n = device_pchk_col_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * n, (int)block_dim.x));
    compute_probs_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<real, false>(device_received_probs),
        ::cuda::matrix_reference<real, false>(device_out_probs),
        ::cuda::matrix_reference<int, false>(device_qmn_row_nxm_indices),
        ::cuda::matrix_reference<real, false>(device_r_mxn),
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // Normalize the computed probabilities.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real, false>(device_out_probs),
        this->clipping_method,
        this->almostzero);
}

template <class GF_q, class real>
__global__ void
hard_decision_kern(
    ::cuda::matrix_reference<real, false> device_out_probs,
    ::cuda::vector_reference<GF_q> received_word,
    basic_hard_decision<real, GF_q, ::cuda::vector_reference<real>>* hd_functor)
{
    int n = received_word.size();
    int pos_n = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos_n < n) {
        received_word(pos_n) =
            (*hd_functor)(device_out_probs.extract_row(pos_n));
    }
}

template <class GF_q, class real>
__global__ void
compute_syndrome_kern(
    ::cuda::vector_reference<int> device_pchk_row_non_zeros,
    ::cuda::matrix_reference<int, false> device_pchk_row_non_zeros_pos,
    ::cuda::matrix_reference<GF_q, false> device_pchk_row_non_zeros_val,
    ::cuda::vector_reference<GF_q> device_received_word,
    ::cuda::vector_reference<GF_q> device_syndrome)
{
    int m = device_syndrome.size();
    int pos_m = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos_m < m) {
        GF_q synd = 0;
        for (int loop_n = 0; loop_n < device_pchk_row_non_zeros(pos_m);
             loop_n++) {
            int pos_n = device_pchk_row_non_zeros_pos(pos_m, loop_n);
            synd += device_pchk_row_non_zeros_val(pos_m, loop_n) *
                    device_received_word(pos_n);
        }

        device_syndrome(pos_m) = synd;
    }
}

template <class GF_q, class real>
__global__ void
check_syndrome_kern(::cuda::vector_reference<GF_q> device_syndrome,
                    bool* decode_success)
{
    bool success = true;

    for (int pos_m = 0; pos_m < device_syndrome.size(); pos_m++)
        success &= !(bool)device_syndrome(pos_m);

    *decode_success = success;
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::decode(libbase::vector<GF_q>& received_word,
                                          int max_iters)
{
    // block size for any kernels called within this function
    int blockdim = 32;

    int n = this->device_received_word.size();
    int m = this->device_syndrome.size();

    bool success;
    for (int curr_cdc_iter = 0; curr_cdc_iter < max_iters; curr_cdc_iter++) {

        // carry out the horizontal step
        // this uses the description of the algorithm as given by
        // MacKay in Information Theory, Inference and Learning Algorithms(2003)
        // on page 560 - chapter 47.3

        // r_mxn(0)=\sum_{x_n'|n'\in N(m)\n'} ( P(z_m=0|x_n=0) * \prod_{n'\in
        // N(m)\n}q_mxn(x_{n') ) Essentially, what we are doing is the
        // following: Assume x_n=0 we need to sum over all possibilities that
        // such that the parity check is satisfied, ie =0 if the parity check is
        // satisfied the conditional probability is 1 and 0 otherwise so we are
        // simply adding up the products for which the parity check is
        // satisfied.
        compute_r_mn();

        // loop over all the symbol nodes - the vertical step
        compute_q_mn();

        // compute the new probabilities for all symbols given the information
        // in this iteration. This will be used in a tentative decoding to see
        // whether we have found a codeword
        compute_probs();

        hard_decision_kern<GF_q, real><<<blockdim, ROUND_UP_DIV(n, blockdim)>>>(
            this->device_out_probs,
            this->device_received_word,
            this->hd_functor.get());
        compute_syndrome_kern<GF_q, real>
            <<<blockdim, ROUND_UP_DIV(m, blockdim)>>>(
                this->device_pchk_row_non_zeros,
                this->device_pchk_row_non_zeros_pos,
                this->device_pchk_row_non_zeros_val,
                this->device_received_word,
                this->device_syndrome);
        check_syndrome_kern<GF_q, real><<<1, 1>>>(
            this->device_syndrome, this->device_decode_success.get());

        this->device_decode_success.to_host(&success);
        if (success)
            break;
    }

    received_word.init(n);
    received_word = (libbase::vector<GF_q>)this->device_received_word;
}

} // namespace libcomm

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// TODO: Add back logrealfast and mpreal
#define REAL_TYPE_SEQ \
      (double)(float)

/* Serialization string: ldpc<type,real>
 * where:
 *      type = gf2 | gf4 ...
 *      real = double | float
 */
#define INSTANTIATE(r, args) \
      template class sum_prod_alg_gdl_cuda<BOOST_PP_SEQ_ENUM(args)>;
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // namespace libcomm
