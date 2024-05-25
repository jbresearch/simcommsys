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
#include "cuda/util.h"
#include "cuda/vector.h"
#include "gf.h"
#include "sum_prod_alg_gdl_cuda.h"
#include "vector.h"
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

/*! \brief Compute ceil(X / Y)
 */
#define ROUND_UP_DIV(X, Y) (((X) + (Y) - 1) / (Y))

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

    // Find the maximum number of non zero elements in a col of the parity
    // check matrix.
    // Also populate pchk_col_non_zeros.
    max_pchk_col_non_zeros = std::numeric_limits<int>::min();
    for (int loop_n = 0; loop_n < n; loop_n++) {
        non_zeros = non_zero_col_pos(loop_n).size();

        pchk_col_non_zeros(loop_n) = non_zeros;
        max_pchk_col_non_zeros = std::max(max_pchk_col_non_zeros, non_zeros);
    }

    // Host fields to build the rest of the parity check matrix repr.
    array1i_t pchk_row_non_zeros_pos(m * max_pchk_row_non_zeros);
    libbase::vector<GF_q> pchk_row_non_zeros_val(m * max_pchk_row_non_zeros);
    libbase::vector<GF_q> pchk_col_non_zeros_val(n * max_pchk_col_non_zeros);

    // we first build mxn_row_idx_lut on the host, then copy to device.
    // Easier since this operation is inherently serial (we have a counter
    // to keep track of current index) and also we need the tanner_edges var
    // computed during this process on host to allocate memory for qmn and
    // rmn matrices.
    array1i_t mx0_row_idx_lut(m);
    array1i_t nxm_row_idx_lut(n * max_pchk_col_non_zeros);

    // counts the number of edges in the Tanner graph of the code.
    // Tells us what the size of device_rmxn and device_qmn_conv should
    // be.
    int tanner_edges = 0;

    // Populate mxn_row_idx_lut
    // Also populate the rest of the parity check matrix repr. on the host.
    // Actual n value, since loop_n is just an index ranging over the number of
    // non-zero values in a row of pchk_matrix.
    int pos_n;
    GF_q val;
    for (int pos_m = 0; pos_m < m; pos_m++) {
        // non-zeros for this row of the parity check matrix
        non_zeros = pchk_row_non_zeros(pos_m);
        mx0_row_idx_lut(pos_m) = tanner_edges;

        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            pos_n = non_zero_row_pos(pos_m)(loop_n) - 1; // we count from zero;
            val = pchk_matrix(pos_m, pos_n);

            // populate other pchk matrix fields on the host.
            pchk_row_non_zeros_pos(pos_m * max_pchk_row_non_zeros + loop_n) =
                pos_n;
            pchk_row_non_zeros_val(pos_m * max_pchk_row_non_zeros + loop_n) =
                val;

            // find loop_m by linear search, should be fast
            int loop_m = 0;
            for (; loop_m < pchk_col_non_zeros(pos_n); loop_m++)
                if (non_zero_col_pos(pos_n)(loop_m) - 1 == pos_m)
                    break;

            // assign an index in device_q_mn_conv, device_r_mxn and so on
            // to a non-zero (m, n) element.
            nxm_row_idx_lut(pos_n * max_pchk_col_non_zeros + loop_m) =
                tanner_edges;
            tanner_edges++;
        }
    }

    int pos_m;
    for (int pos_n = 0; pos_n < n; pos_n++) {
        // non-zeros for this row of the parity check matrix
        non_zeros = pchk_col_non_zeros(pos_n);

        for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
            pos_m = non_zero_col_pos(pos_n)(loop_m) - 1; // we count from zero;
            val = pchk_matrix(pos_m, pos_n);

            pchk_col_non_zeros_val(pos_n * max_pchk_col_non_zeros + loop_m) =
                val;
        }
    }

    device_mx0_row_idx_lut.init(m);
    device_nxm_row_idx_lut.init(n, max_pchk_col_non_zeros);
    // Copy mxn_row_idx_lut to device
    device_mx0_row_idx_lut = mx0_row_idx_lut;
    device_nxm_row_idx_lut = nxm_row_idx_lut;

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
}

// Inspired by
// https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_64_website/samples.html#fastWalshTransform
template <class GF_q, class real>
__global__ void
hadamard_transform_kern(::cuda::matrix_reference<real> src,
                        int tanner_edges,
                        int hmax)
{
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(real)) unsigned char sbuf[];
    real* sdata = reinterpret_cast<real*>(sbuf);

    int tid_mod_hmax = threadIdx.x & (hmax - 1);
    int i1 = 2 * (threadIdx.x - tid_mod_hmax) + tid_mod_hmax;
    int i2 = i1 + hmax;

    int bid_mod_hmax = (blockIdx.x * blockDim.x) & (hmax - 1);
    int bi = 2 * (blockIdx.x * blockDim.x - bid_mod_hmax) + bid_mod_hmax;

    int num_of_elements = GF_q::elements();
    int log2_num_of_elements = GF_q::log2_elements();

    int loop_i = (bi + i1) >> log2_num_of_elements;
    int loop_e = (bi + i1) & (num_of_elements - 1);

    if (loop_i < tanner_edges) {
        sdata[i1] = src(loop_i, loop_e);
        sdata[i2] = src(loop_i, loop_e + hmax);
        __syncthreads();

        for (int h = 1; h <= hmax; h <<= 1) {
            int tid_mod_h = threadIdx.x & (h - 1);
            int i1 = 2 * (threadIdx.x - tid_mod_h) + tid_mod_h;
            int i2 = i1 + h;

            real tmp1 = sdata[i1];
            real tmp2 = sdata[i2];

            sdata[i1] = tmp1 + tmp2;
            sdata[i2] = tmp1 - tmp2;

            __syncthreads();
        }

        src(loop_i, loop_e) = sdata[i1];
        src(loop_i, loop_e + hmax) = sdata[i2];
    }
}

template <class GF_q, class real>
__global__ void
hadamard_transform_pass_kern(::cuda::matrix_reference<real> src,
                             ::cuda::matrix_reference<real> dst,
                             int tanner_edges,
                             int h)
{
    // find loop_e
    int loop_e = blockIdx.x * blockDim.x + threadIdx.x;

    // this is just a generic index that ranges over [0, tanner_edges)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
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
inline void
hadamard_transform(::cuda::matrix_reference<real>& src,
                   ::cuda::matrix_reference<real>& dst)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    int num_of_elements = GF_q::elements();
    int tanner_edges = src.get_rows();

    int device = ::cuda::cudaGetCurrentDevice();
    int max_threads_per_block = ::cuda::cudaGetMaxThreadsPerBlock(device);
    int max_smem_per_block = ::cuda::cudaGetSharedMemPerBlock(device);

    dim3 block_dim = dim3(min(max_threads_per_block,
                              max_smem_per_block / (int)(2 * sizeof(real))));
    // use division which truncates upwards.
    // divide num_of_elements by 2 as each block processes two elements.
    dim3 num_blocks = dim3(
        ROUND_UP_DIV((num_of_elements >> 1) * tanner_edges, (int)block_dim.x));

    int hmax = min(num_of_elements >> 1, block_dim.x);
    hadamard_transform_kern<GF_q, real>
        <<<num_blocks, block_dim, 2 * sizeof(real) * block_dim.x>>>(
            src, tanner_edges, hmax);
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements, (int)block_dim.x),
                      ROUND_UP_DIV(tanner_edges, (int)block_dim.y));

    for (int h = hmax << 1; h < num_of_elements; h <<= 1) {
        hadamard_transform_pass_kern<GF_q, real>
            <<<num_blocks, block_dim>>>(src, dst, tanner_edges, h);
        cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
        cudaDeviceSynchronize();
#endif

        std::swap(src, dst);
    }
}

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
clip_and_normalize_probs_kern(::cuda::matrix_reference<real> probs,
                              int clipping_method,
                              real almost_zero)
{
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(real)) unsigned char sbuf[];
    real* sdata = reinterpret_cast<real*>(sbuf);

    // ranges over probability distributions in prob.
    int loop_n = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds checking
    int n = probs.get_rows();
    if (loop_n < n) {
        int i = threadIdx.y * blockDim.x + threadIdx.x;
        sdata[i] = 0.0;

        int num_of_elements = GF_q::elements();

        real tmp_prob;
        for (int loop_e = threadIdx.x; loop_e < num_of_elements;
             loop_e += blockDim.x) {
            // Clipping HACK
            tmp_prob = probs(loop_n, loop_e);
            perform_clipping(tmp_prob, clipping_method, almost_zero);
            probs(loop_n, loop_e) = tmp_prob;
            sdata[i] += tmp_prob;
        }

        __syncthreads();

        real alpha = 0.0;
        // linear reduce of partial sums, fast
        for (int i = 0; i < blockDim.x; i++)
            alpha += sdata[i];

        cuda_assertalways(alpha != real(0.0));

        // normalize probabilities (divide by alpha)
        for (int loop_e = threadIdx.x; loop_e < num_of_elements;
             loop_e += blockDim.x) {
            probs(loop_n, loop_e) /= alpha;
        }
    }
}

template <class GF_q, class real>
__global__ void
clip_and_normalize_probs_fast_kern(::cuda::matrix_reference<real> probs,
                                   int clipping_method,
                                   real almost_zero)
{
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(real)) unsigned char sbuf[];
    real* psums = reinterpret_cast<real*>(sbuf);
    real* sdata = reinterpret_cast<real*>(sbuf) + blockDim.x;

    constexpr int num_of_elements = GF_q::elements();

    int loop_e = threadIdx.x;
    int loop_n = blockIdx.x * blockDim.x;

    int n = probs.get_rows();
    if (loop_n < n) {
        sdata[threadIdx.x] = probs(loop_n, loop_e);
        perform_clipping(sdata[threadIdx.x], clipping_method, almost_zero);

        psums[threadIdx.x] = sdata[threadIdx.x];
        __syncthreads();

        for (int stride = 1; stride < num_of_elements; stride <<= 1) {
            // NOTE: reads may be out of bounds but we will just end up in sdata
            // buffer
            psums[threadIdx.x] += psums[threadIdx.x + stride];
            __syncthreads();
        }

        sdata[threadIdx.x] /= psums[0];
        probs(loop_n, loop_e) = sdata[threadIdx.x];
    }
}

template <class GF_q, class real>
inline void
clip_and_normalize_probs(::cuda::matrix_reference<real> probs,
                         int clipping_method,
                         real almost_zero)
{
    int n = probs.get_rows();
    int num_of_elements = GF_q::elements();

    int device = ::cuda::cudaGetCurrentDevice();
    int warpsize = ::cuda::cudaGetWarpSize(device);

    dim3 block_dim, num_blocks;

    if (num_of_elements < warpsize) {
        int max_threads_per_block = ::cuda::cudaGetMaxThreadsPerBlock(device);
        int shared_mem_per_block = ::cuda::cudaGetSharedMemPerBlock(device);

        block_dim = dim3(
            warpsize,
            min(max_threads_per_block / warpsize,
                shared_mem_per_block / (int)(2 * sizeof(real) * warpsize)));
        num_blocks = dim3(1, ROUND_UP_DIV(n, (int)block_dim.y));

        clip_and_normalize_probs_kern<GF_q, real>
            <<<num_blocks,
               block_dim,
               2 * block_dim.x * block_dim.y * sizeof(real)>>>(
                probs, clipping_method, almost_zero);
    } else {
        block_dim = dim3(num_of_elements);
        num_blocks = dim3(ROUND_UP_DIV(n, (int)block_dim.y));

        clip_and_normalize_probs_fast_kern<GF_q, real>
            <<<num_blocks, block_dim, 2 * sizeof(real) * block_dim.x>>>(
                probs, clipping_method, almost_zero);
    }

    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
}

template <class GF_q, class real>
__global__ void
spa_init_kern(::cuda::matrix_reference<real> device_received_probs,
              ::cuda::vector_reference<int> device_mx0_row_idx_lut,
              ::cuda::matrix_reference<real> device_r_mxn,
              ::cuda::matrix_reference<real> device_qmn_conv,
              ::cuda::vector_reference<int> device_pchk_row_non_zeros,
              ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
              ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val)
{
    int num_of_elements = GF_q::elements();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // find loop_e
    int loop_e = i % num_of_elements;
    // find loop_m
    int loop_m = i / num_of_elements;

    // bounds checking
    int m = device_pchk_row_non_zeros.size();
    if (loop_m < m) {
        int non_zeros = device_pchk_row_non_zeros(loop_m);

        int qmn_row_idx;
        int pos;
        GF_q h_m_n;
        // NOTE: loop_n iterates over number of symbols that participate in
        // mth check of a codeword. E.g. if check involves {x_1, x_4, x_6},
        // loop_n ranges over [0, 1, 2]
        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            // NOTE: pos is the actual index of the nth symbol participating
            // in the mth check in the codeword. E.g. if check involves
            // {x_1, x_4, x_6} and loop_n = 1, pos = 4 (-1 since we count
            // from 0)
            pos = device_pchk_row_non_zeros_pos(loop_m, loop_n);
            // NOTE: Find corresponding value in the parity check matrix.
            // We use loop_m because this is the check index, and pos
            // because this is the actual index of the nth symbol
            // participating in the mth check (non_zeros variable does not
            // count symbols that don't participate in the mth check).
            h_m_n = device_pchk_row_non_zeros_val(loop_m, loop_n);

            // get index into device_qmn_conv and device_r_mxn
            qmn_row_idx = device_mx0_row_idx_lut(loop_m) + loop_n;

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
                device_received_probs(pos, loop_e);
        }
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

    // Convert vector of vectors into a single vector so that it can be
    // copied to device more efficiently
    array1d_t recvd_probs_flat(dim_n * num_of_elements);
    for (int loop_n = 0; loop_n < dim_n; loop_n++)
        for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
            recvd_probs_flat(loop_n * num_of_elements + loop_e) =
                recvd_probs(loop_n)(loop_e);

    this->device_received_probs = recvd_probs_flat;

    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real>(this->device_received_probs),
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

    int m = this->device_pchk_row_non_zeros.size();
    block_dim = dim3(512);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * m, (int)block_dim.x));
    spa_init_kern<GF_q, real>
        <<<num_blocks, block_dim>>>(this->device_received_probs,
                                    this->device_mx0_row_idx_lut,
                                    this->device_r_mxn,
                                    this->device_qmn_conv,
                                    this->device_pchk_row_non_zeros,
                                    this->device_pchk_row_non_zeros_pos,
                                    this->device_pchk_row_non_zeros_val);
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // apply the FFT again to get the proper values
    // Here we use matrix references for cheap swapping. The result of the
    // Hadamard transform will always be in src.
    ::cuda::matrix_reference<real> src(device_qmn_conv);
    ::cuda::matrix_reference<real> dst(device_swap_buf);
    hadamard_transform<GF_q, real>(src, dst);

    // Result of the Hadamard transform is always stored in first arg passed
    // to hadamard_transform(), copy to device_qmn_conv in case src is the
    // swap buffer.
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
compute_r_mn_kern(::cuda::vector_reference<int> device_mx0_row_idx_lut,
                  ::cuda::matrix_reference<real> device_r_mxn,
                  ::cuda::matrix_reference<real> device_qmn_conv,
                  ::cuda::vector_reference<int> device_pchk_row_non_zeros)
{
    int num_of_elements = GF_q::elements();
    int m = device_pchk_row_non_zeros.size();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int loop_e = i % num_of_elements;
    int loop_m = i / num_of_elements;

    if (loop_m < m) {
        int non_zeros = device_pchk_row_non_zeros(loop_m);
        // Holds the actual message computed
        real q_nm_conv_prod = 1.0;
        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            // NOTE: Branchless computation
            q_nm_conv_prod *= device_qmn_conv(
                device_mx0_row_idx_lut(loop_m) + loop_n, loop_e);
        }

        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            int row_idx = device_mx0_row_idx_lut(loop_m) + loop_n;
            // coalesced memory access due to syncthreads above
            // We store in r_mxn but this is not the final result.
            device_r_mxn(row_idx, loop_e) =
                q_nm_conv_prod / device_qmn_conv(row_idx, loop_e);
        }
    }
}

template <class GF_q, class real>
void
compute_r_mn(::cuda::vector<int>& device_mx0_row_idx_lut,
             ::cuda::matrix<real>& device_r_mxn,
             ::cuda::matrix<real>& device_qmn_conv,
             ::cuda::vector<int>& device_pchk_row_non_zeros,
             ::cuda::matrix<GF_q>& device_pchk_row_non_zeros_val,
             ::cuda::matrix<real>& device_swap_buf,
             int clipping_method,
             real almost_zero)
{
    dim3 block_dim, num_blocks;

    int m = device_pchk_row_non_zeros.size();
    int num_of_elements = GF_q::elements();

    int device = ::cuda::cudaGetCurrentDevice();
    int warpsize = ::cuda::cudaGetWarpSize();

    block_dim = dim3(warpsize);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * m, (int)block_dim.x));
    compute_r_mn_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::vector_reference<int>(device_mx0_row_idx_lut),
        ::cuda::matrix_reference<real>(device_r_mxn),
        ::cuda::matrix_reference<real>(device_qmn_conv),
        ::cuda::vector_reference<int>(device_pchk_row_non_zeros));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // apply the FFT again to get the proper values
    // Here we use matrix references for cheap swapping.
    ::cuda::matrix_reference<real> src(device_r_mxn);
    ::cuda::matrix_reference<real> dst(device_swap_buf);
    hadamard_transform<GF_q, real>(src, dst);

    // dst could be device_r_mxn or device_swap_buf depending on whether no.
    // of passes in Hadamard transform is even or odd. We copy back to
    // device_r_mxn to make sure the result is in the right array.
    device_r_mxn = src;

    // Apply clipping + normalization to the computed r_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real>(device_r_mxn),
        clipping_method,
        almost_zero);
}

template <class GF_q, class real>
__global__ void
compute_q_mn_kern(::cuda::matrix_reference<real> device_received_probs,
                  ::cuda::matrix_reference<int> device_nxm_row_idx_lut,
                  ::cuda::matrix_reference<real> device_r_mxn,
                  ::cuda::matrix_reference<real> device_qmn_conv,
                  ::cuda::vector_reference<int> device_pchk_col_non_zeros,
                  ::cuda::matrix_reference<GF_q> device_pchk_col_non_zeros_val)
{
    // find loop_e
    int loop_e = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

    // find loop_n
    int loop_n = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    loop_n = min(loop_n, n - 1);

    // Holds the actual message computed
    real q_nm = device_received_probs(loop_n, loop_e);

    int non_zeros = device_pchk_col_non_zeros(loop_n);
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        GF_q h_m_n = device_pchk_col_non_zeros_val(loop_n, loop_m);
        // NOTE: Branchless computation
        q_nm *= device_r_mxn(device_nxm_row_idx_lut(loop_n, loop_m),
                             GF_q(loop_e) * h_m_n);
    }
    // Loop above has potential divergence as different m have different
    // degrees in general. We want to convergence again here so most iters
    // are in sync.
    // TODO: Test impact of this.
    __syncthreads();

    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        GF_q h_m_n = device_pchk_col_non_zeros_val(loop_n, loop_m);

        int row_idx = device_nxm_row_idx_lut(loop_n, loop_m);
        // Uncoalesced memory access.
        device_qmn_conv(row_idx, GF_q(loop_e) * h_m_n) =
            q_nm / device_r_mxn(row_idx, GF_q(loop_e) * h_m_n);
    }
}

template <class GF_q, class real>
void
compute_q_mn(::cuda::matrix<real>& device_received_probs,
             ::cuda::vector<int>& device_mx0_row_idx_lut,
             ::cuda::matrix<int>& device_nxm_row_idx_lut,
             ::cuda::matrix<real>& device_r_mxn,
             ::cuda::matrix<real>& device_qmn_conv,
             ::cuda::vector<int>& device_pchk_row_non_zeros,
             ::cuda::matrix<GF_q>& device_pchk_row_non_zeros_val,
             ::cuda::vector<int>& device_pchk_col_non_zeros,
             ::cuda::matrix<GF_q>& device_pchk_col_non_zeros_val,
             ::cuda::matrix<real>& device_swap_buf,
             int clipping_method,
             real almost_zero)
{

    dim3 block_dim, num_blocks;

    int n = device_pchk_col_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements, (int)block_dim.x),
                      ROUND_UP_DIV(n, (int)block_dim.y));
    compute_q_mn_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<real>(device_received_probs),
        ::cuda::matrix_reference<int>(device_nxm_row_idx_lut),
        ::cuda::matrix_reference<real>(device_r_mxn),
        ::cuda::matrix_reference<real>(device_qmn_conv),
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros),
        ::cuda::matrix_reference<GF_q>(device_pchk_col_non_zeros_val));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // Here we use matrix references for cheap swapping.
    ::cuda::matrix_reference<real> src(device_qmn_conv);
    ::cuda::matrix_reference<real> dst(device_swap_buf);

    // Compute Hadamard transform on the result.
    hadamard_transform<GF_q, real>(src, dst);

    // Result of the Hadamard transform is always stored in first arg passed
    // to hadamard_transform(), copy to device_qmn_conv in case dst is the
    // swap buffer.
    device_qmn_conv = src;
}

template <class GF_q, class real>
__global__ void
compute_probs_kern(::cuda::matrix_reference<real> device_received_probs,
                   ::cuda::matrix_reference<real> device_out_probs,
                   ::cuda::matrix_reference<int> device_nxm_row_idx_lut,
                   ::cuda::matrix_reference<real> device_r_mxn,
                   ::cuda::vector_reference<int> device_pchk_col_non_zeros,
                   ::cuda::matrix_reference<GF_q> device_pchk_col_non_zeros_val)
{
    // find loop_e
    int loop_e = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

    // find loop_n
    int loop_n = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    loop_n = min(loop_n, n - 1);

    int non_zeros = device_pchk_col_non_zeros(loop_n);
    GF_q h_m_n;
    // Holds the prob computed
    real prob = device_received_probs(loop_n, loop_e);
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        h_m_n = device_pchk_col_non_zeros_val(loop_n, loop_m);

        prob *= device_r_mxn(device_nxm_row_idx_lut(loop_n, loop_m),
                             h_m_n * GF_q(loop_e));
    }

    device_out_probs(loop_n, loop_e) = prob;
}

template <class GF_q, class real>
void
compute_probs(::cuda::matrix<real>& device_received_probs,
              ::cuda::matrix<real>& device_out_probs,
              ::cuda::matrix<int>& device_nxm_row_idx_lut,
              ::cuda::matrix<real>& device_r_mxn,
              ::cuda::vector<int>& device_pchk_col_non_zeros,
              ::cuda::matrix<GF_q>& device_pchk_col_non_zeros_val,
              int clipping_method,
              real almost_zero)
{
    dim3 block_dim, num_blocks;

    int n = device_pchk_col_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements, (int)block_dim.x),
                      ROUND_UP_DIV(n, (int)block_dim.y));
    compute_probs_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<real>(device_received_probs),
        ::cuda::matrix_reference<real>(device_out_probs),
        ::cuda::matrix_reference<int>(device_nxm_row_idx_lut),
        ::cuda::matrix_reference<real>(device_r_mxn),
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros),
        ::cuda::matrix_reference<GF_q>(device_pchk_col_non_zeros_val));
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    // Normalize the computed probabilities.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real>(device_out_probs),
        clipping_method,
        almost_zero);
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::spa_iteration(array1vd_t& ro)
{
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

    compute_r_mn(this->device_mx0_row_idx_lut,
                 this->device_r_mxn,
                 this->device_qmn_conv,
                 this->device_pchk_row_non_zeros,
                 this->device_pchk_row_non_zeros_val,
                 this->device_swap_buf,
                 this->clipping_method,
                 this->almostzero);

    // loop over all the symbol nodes - the vertical step

    compute_q_mn(this->device_received_probs,
                 this->device_mx0_row_idx_lut,
                 this->device_nxm_row_idx_lut,
                 this->device_r_mxn,
                 this->device_qmn_conv,
                 this->device_pchk_row_non_zeros,
                 this->device_pchk_row_non_zeros_val,
                 this->device_pchk_col_non_zeros,
                 this->device_pchk_col_non_zeros_val,
                 this->device_swap_buf,
                 this->clipping_method,
                 this->almostzero);

    // compute the new probabilities for all symbols given the information
    // in this iteration. This will be used in a tentative decoding to see
    // whether we have found a codeword
    compute_probs<GF_q, real>(this->device_received_probs,
                              this->device_out_probs,
                              this->device_nxm_row_idx_lut,
                              this->device_r_mxn,
                              this->device_pchk_col_non_zeros,
                              this->device_pchk_col_non_zeros_val,
                              this->clipping_method,
                              this->almostzero);

    // Copy received probabilities from device to host.
    array1d_t ro_m = (::libbase::vector<real>)this->device_out_probs;

    // ensure ro has the right size
    ro.init(this->device_out_probs.get_rows());

    int cols = this->device_out_probs.get_cols();
    for (int n = 0; n < ro.size(); n++) {
        // allocate memory on host for probability distribution of symbol n
        ro(n).init(cols);
        for (int loop_e = 0; loop_e < ro(n).size(); loop_e++)
            ro(n)(loop_e) = ro_m(n * cols + loop_e);
    }
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
