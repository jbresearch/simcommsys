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
#include "cuda/device_ptr.h"
#include "cuda/gputimer.h"
#include "cuda/matrix.h"
#include "cuda/stream.h"
#include "cuda/vector.h"
#include "gf.h"
#include "hard_decision.h"
#include "sum_prod_alg_gdl_cuda.h"
#include "vector.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

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
    seed_hd_functor<<<1, 1>>>(this->hd_functor.get(), r.ival());
}

/*! \brief Compute ceil(X / Y)
 */
#define ROUND_UP_DIV(X, Y) (((X) + (Y) - 1) / (Y))

enum PermutationType { MULTIPLY, DIVIDE };

template <class GF_q, class real, PermutationType permtype>
__global__ void
hadamard_transform_kern(
    ::cuda::matrix_reference<real, false> hadamard_buf,
    ::cuda::vector_reference<GF_q> device_pchk_non_zeros_val)
{
    // Declaring a type-parametrized extern symbol in a template function
    // will cause a name conflict if the template is instantiated multiple
    // times. This is a problem since dynamically sized shared memory in
    // CUDA is an extern symbol. So we declare a buffer of char aligned to
    // the required type and then cast to a pointer of the type parameter.
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(real)) char rawbuf[];
    real* buf = reinterpret_cast<real*>(rawbuf);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // index over row in hadamard_buf
    int pos_r = idx / GF_q::elements();
    int r = hadamard_buf.get_rows();

    if (pos_r < r) {
        int loop_e = idx % GF_q::elements();
        buf[threadIdx.x] = hadamard_buf(pos_r, loop_e);
    }
    __syncthreads();

    if (permtype == MULTIPLY) {
        if (pos_r < r) {
            int loop_e = idx % GF_q::elements();
            GF_q h_m_n = device_pchk_non_zeros_val(pos_r);

            int offset = threadIdx.x & ~(GF_q::elements() - 1);
            buf[offset + h_m_n * GF_q(loop_e)] = buf[threadIdx.x];
        }
        __syncthreads();
    }

    for (int h = 1; h < GF_q::elements(); h <<= 1) {
        int loop_e = idx % GF_q::elements();

        // If floor(loop_e / h) is odd, sign is -1.0
        // If floor(loop_e / h) is even, sign is 1.0
        int sign = ((real)((loop_e / h) % 2 == 0) - 0.5) * 2.0;

        // From the butterfly property:
        // If floor(loop_e / h) is odd, result of the pass is P[loop_e - h] -
        // P[e] If floor(loop_e / h) is even, result of the pass is P[loop_e +
        // h] + P[e]
        buf[threadIdx.x] =
            buf[int(threadIdx.x) + sign * h] + sign * buf[threadIdx.x];
        __syncthreads();
    }

    if (permtype == DIVIDE) {
        if (pos_r < r) {
            int loop_e = idx % GF_q::elements();
            GF_q h_m_n = device_pchk_non_zeros_val(pos_r);

            int offset = threadIdx.x & ~(GF_q::elements() - 1);
            buf[threadIdx.x] = buf[offset + h_m_n * GF_q(loop_e)];
        }
        __syncthreads();
    }

    if (pos_r < r) {
        int loop_e = idx % GF_q::elements();
        hadamard_buf(pos_r, loop_e) = buf[threadIdx.x];
    }
}

template <class GF_q, class real, PermutationType permtype>
inline void
hadamard_transform(::cuda::matrix_reference<real, false> hadamard_buf,
                   ::cuda::vector_reference<GF_q> device_pchk_non_zeros_val)
{
    int tanner_edges = hadamard_buf.get_rows();

    int max_threads_per_block =
        ::cuda::cudaGetMaxThreadsPerBlock(::cuda::cudaGetCurrentDevice());

    int smem_per_block =
        ::cuda::cudaGetSharedMemPerBlock(::cuda::cudaGetCurrentDevice());
    int block_dim =
        std::min(max_threads_per_block, smem_per_block / int(sizeof(real)));

    // Hadamard transform over a single row must always fit in a block.
    assertalways(block_dim >= GF_q::elements());

    int num_blocks =
        ROUND_UP_DIV(tanner_edges * GF_q::elements(), int(block_dim));

    hadamard_transform_kern<GF_q, real, permtype>
        <<<num_blocks, block_dim, block_dim * sizeof(real)>>>(
            hadamard_buf, device_pchk_non_zeros_val);
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
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

    // overallocate, but that's fine, final segment is not used
    libbase::vector<GF_q> pchk_non_zeros_val(n * max_pchk_col_non_zeros);

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
            pchk_non_zeros_val(tanner_edges) = pchk_matrix(pos_m, pos_n);

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

    // Copy qmn_row_nxm_indices, qmn_row_nxm_indices to device
    device_qmn_row_nxm_indices = qmn_row_nxm_indices;
    device_qmn_row_mxn_indices = qmn_row_mxn_indices;

    // Copy represenation of the parity check matrix to the device.
    device_pchk_row_non_zeros = pchk_row_non_zeros;
    device_pchk_row_non_zeros_pos = pchk_row_non_zeros_pos;
    device_pchk_row_non_zeros_val = pchk_row_non_zeros_val;

    device_pchk_col_non_zeros = pchk_col_non_zeros;
    device_pchk_non_zeros_val = pchk_non_zeros_val;

    // Allocate required memory for r_mxn, q_mxn and qmn_conv on device.
    device_r_mxn.init(tanner_edges, num_of_elements);
    device_qmn_conv.init(tanner_edges, num_of_elements);

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
    // compute alpha
    // Declaring a type-parametrized extern symbol in a template function
    // will cause a name conflict if the template is instantiated multiple
    // times. This is a problem since dynamically sized shared memory in
    // CUDA is an extern symbol. So we declare a buffer of char aligned to
    // the required type and then cast to a pointer of the type parameter.
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(real)) char psums_buf[];
    real* psums = reinterpret_cast<real*>(psums_buf);

    int num_of_elements = GF_q::elements();
    int num_of_elements_div_2 = num_of_elements / 2;

    // each block processes 2 * blockDim.x elements, 2 per thread.
    // If we lay out all elements accessed by (loop_n, loop_e) in row-major
    // order, it is not difficult to see that i0, i1 are the indices handled by
    // this thread:
    int i0 = int(2 * blockIdx.x * blockDim.x + threadIdx.x);
    int i1 = i0 + blockDim.x;

    int loop_n0 = i0 / num_of_elements;
    int loop_n1 = i1 / num_of_elements;

    int n = probs.get_rows();

    // load probabilities for this thread and clip them.
    real prob0 = 0;
    if (loop_n0 < n) {
        int loop_e0 = i0 % num_of_elements;

        prob0 = probs(loop_n0, loop_e0);
        perform_clipping(prob0, clipping_method, almostzero);
    }

    real prob1 = 0;
    if (loop_n1 < n) {
        int loop_e1 = i1 % num_of_elements;

        prob1 = probs(loop_n1, loop_e1);
        perform_clipping(prob1, clipping_method, almostzero);
    }

    psums[threadIdx.x] = prob0;
    psums[threadIdx.x + blockDim.x] = prob1;
    __syncthreads();

    // (almost) divergence free, parallel optimized summation.
    // NOTE: GF_q::elements() is used instead of num_of_elements to
    // encourage loop unrolling
    for (int stride = 1; stride < GF_q::elements(); stride *= 2) {
        if (threadIdx.x < int(blockDim.x) / stride) {
            psums[threadIdx.x * 2 * stride] +=
                psums[threadIdx.x * 2 * stride + stride];
        }

        __syncthreads();
    }

    real alpha0 =
        psums[(threadIdx.x >> GF_q::log2_elements()) << GF_q::log2_elements()];
    real alpha1 = psums[((threadIdx.x + blockDim.x) >> GF_q::log2_elements())
                        << GF_q::log2_elements()];

    // normalize probabilities (divide by alpha)
    if (loop_n0 < n) {
        int loop_e0 = i0 % num_of_elements;

        cuda_assertalways(alpha0 != real(0.0));
        probs(loop_n0, loop_e0) = prob0 / alpha0;
    }
    if (loop_n1 < n) {
        int loop_e1 = i1 % num_of_elements;

        cuda_assertalways(alpha1 != real(0.0));
        probs(loop_n1, loop_e1) = prob1 / alpha1;
    }
}

template <class GF_q, class real>
inline void
clip_and_normalize_probs(::cuda::matrix_reference<real, false> probs,
                         int clipping_method,
                         real almostzero)
{
    int n = probs.get_rows();

    int max_threads_per_block =
        ::cuda::cudaGetMaxThreadsPerBlock(::cuda::cudaGetCurrentDevice());

    int smem_per_block =
        ::cuda::cudaGetSharedMemPerBlock(::cuda::cudaGetCurrentDevice());
    int block_dim =
        std::min(max_threads_per_block, smem_per_block / int(2 * sizeof(real)));

    int num_blocks = ROUND_UP_DIV(n * GF_q::elements(), int(2 * block_dim));

    // summation of probabilities over a single row must always fit in a
    // block.
    assertalways(2 * block_dim >= GF_q::elements());

    clip_and_normalize_probs_kern<GF_q, real>
        <<<num_blocks, block_dim, 2 * block_dim * sizeof(real)>>>(
            probs, clipping_method, almostzero);
    cudaSafeCall(cudaGetLastError());

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
}

template <class GF_q, class real>
__global__ void
spa_init_kern(::cuda::matrix_reference<real, false> device_received_probs,
              ::cuda::matrix_reference<int, false> device_qmn_row_nxm_indices,
              ::cuda::matrix_reference<real, false> device_r_mxn,
              ::cuda::matrix_reference<real, false> device_qmn_conv,
              ::cuda::vector_reference<int> device_pchk_col_non_zeros)
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
    // NOTE: loop_m iterates over the number of checks that symbol pos_n
    // participates in.
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        // get index into device_qmn_conv and device_r_mxn
        qmn_row_idx = device_qmn_row_nxm_indices(pos_n, loop_m);

        device_qmn_conv(qmn_row_idx, loop_e) =
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

    ::cuda::gputimer t_spa_init_copy_probs("t__spa_init__copy_probs_h_to_d");

    for (int loop_n = 0; loop_n < dim_n; loop_n++)
        this->device_received_probs.extract_row(loop_n) = recvd_probs(loop_n);

    this->add_timer(t_spa_init_copy_probs);

    ////// BEGIN NORMALIZE
    ::cuda::gputimer t_spa_init_norm_probs("t__spa_init__norm_probs");

    clip_and_normalize_probs<GF_q, real>(
        this->device_received_probs, this->clipping_method, this->almostzero);

    this->add_timer(t_spa_init_norm_probs);
    ////// END NORMALIZE

    // this uses the description of the algorithm as given by

    // MacKay in Information Theory, Inference and Learning Algorithms(2003)
    // on page 560 - chapter 47.3

    // some helper variables

    ////// BEGIN SPA INIT KERN
    ::cuda::gputimer t_spa_init_kern("t__spa_init__spa_init_kern");

    block_dim = dim3(1024);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(num_of_elements * dim_n, (int)block_dim.x));
    spa_init_kern<GF_q, real>
        <<<num_blocks, block_dim>>>(this->device_received_probs,
                                    this->device_qmn_row_nxm_indices,
                                    this->device_r_mxn,
                                    this->device_qmn_conv,
                                    this->device_pchk_col_non_zeros);
    cudaSafeCall(cudaGetLastError());

    this->add_timer(t_spa_init_kern);

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
    ////// END SPA INIT KERN

    ////// BEGIN HADAMARD TRANSFORM
    ::cuda::gputimer t_spa_init_hadamard("t__spa_init__hadamard");

    // apply the FFT again to get the proper values
    hadamard_transform<GF_q, real, MULTIPLY>(device_qmn_conv,
                                             device_pchk_non_zeros_val);

    this->add_timer(t_spa_init_hadamard);
    ////// END HADAMARD TRANSFORM

    this->decode_success = false;
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

    ////// BEGIN COMPUTE R_MN
    ::cuda::gputimer t_compute_r_mn("t_compute_r_mn");

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

    this->add_timer(t_compute_r_mn);

#ifdef DEBUG
    cudaDeviceSynchronize();
#endif
    ////// END COMPUTE R_MN

    ////// BEGIN INVERSE HADAMARD
    ::cuda::gputimer t_inv_hadamard("t_inv_hadamard");
    // apply the FFT again to get the proper values
    hadamard_transform<GF_q, real, DIVIDE>(device_r_mxn,
                                           device_pchk_non_zeros_val);

    this->add_timer(t_inv_hadamard);
    ////// END INVERSE HADAMARD

    ////// BEGIN NORMALIZE
    ::cuda::gputimer t_norm_r_mn("t_norm_r_mn");

    // Apply clipping + normalization to the computed r_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real, false>(device_r_mxn),
        this->clipping_method,
        this->almostzero);

    this->add_timer(t_norm_r_mn);
    ////// END NORMALIZE
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

    ////// BEGIN COMPUTE Q_MN
    ::cuda::gputimer t_compute_q_mn("t_compute_q_mn");

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

    this->add_timer(t_compute_q_mn);
    ////// END COMPUTE Q_MN

    ////// BEGIN NORMALIZE
    ::cuda::gputimer t_norm_q_mn("t_norm_q_mn");

    // Apply clipping + normalization to the computed q_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real, false>(device_qmn_conv),
        this->clipping_method,
        this->almostzero);

    this->add_timer(t_norm_q_mn);
    ////// END NORMALIZE

    ////// BEGIN HADAMARD TRANSFORM
    ::cuda::gputimer t_hadamard("t_hadamard");

    // Compute Hadamard transform on the result.
    hadamard_transform<GF_q, real, MULTIPLY>(device_qmn_conv,
                                             device_pchk_non_zeros_val);

    this->add_timer(t_hadamard);
    ////// END HADAMARD TRANSFORM
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

    ////// BEGIN COMPUTE PROBS
    ::cuda::gputimer t_compute_probs("t_compute_probs");

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

    this->add_timer(t_compute_probs);
    ////// END COMPUTE PROBS

    ////// BEGIN NORMALIZE PROBS
    ::cuda::gputimer t_norm_probs("t_norm_probs");

    // Normalize the computed probabilities.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real, false>(device_out_probs),
        this->clipping_method,
        this->almostzero);

    this->add_timer(t_norm_probs);
    ////// END NORMALIZE PROBS
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
bool
sum_prod_alg_gdl_cuda<GF_q, real>::spa_iteration()
{
    // block size for any kernels called within this function
    int blockdim = 32;
    int n = this->device_received_word.size();
    int m = this->device_syndrome.size();

    bool success;

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

    ::cuda::gputimer t_hard_decision("t_hard_decision");

    hard_decision_kern<GF_q, real>
        <<<blockdim, ROUND_UP_DIV(n, blockdim)>>>(this->device_out_probs,
                                                  this->device_received_word,
                                                  this->hd_functor.get());

    this->add_timer(t_hard_decision);

    ::cuda::gputimer t_compute_syndrome("t_compute_syndrome");

    compute_syndrome_kern<GF_q, real><<<blockdim, ROUND_UP_DIV(m, blockdim)>>>(
        this->device_pchk_row_non_zeros,
        this->device_pchk_row_non_zeros_pos,
        this->device_pchk_row_non_zeros_val,
        this->device_received_word,
        this->device_syndrome);

    this->add_timer(t_compute_syndrome);

    ::cuda::gputimer t_check_syndrome("t_check_syndrome");

    check_syndrome_kern<GF_q, real>
        <<<1, 1>>>(this->device_syndrome, this->device_decode_success.get());

    this->add_timer(t_check_syndrome);
    this->device_decode_success.to_host(&success);

    return success;
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::spa_iteration(
    libbase::vector<GF_q>& received_word)
{
    ::cuda::gputimer t_spa_iteration("t_spa_iteration");

    if (this->decode_success) { // codeword was found in previous iteration
        received_word = this->received_word;
    } else {
        if (spa_iteration()) { // this was the last iteration; we found a
                               // codeword
            // Copy the received codeword from the GPU.
            ::cuda::gputimer t_copy_codeword("t_copy_codeword_d_to_h");
            received_word = this->received_word =
                (libbase::vector<GF_q>)this->device_received_word;
            this->add_timer(t_copy_codeword);

            this->decode_success = true;
        } else {
            received_word = (libbase::vector<GF_q>)this->device_received_word;
        }
    }

    this->add_timer(t_spa_iteration);
}

template <class GF_q, class real>
void
sum_prod_alg_gdl_cuda<GF_q, real>::decode(libbase::vector<GF_q>& received_word,
                                          int max_iters)
{
    bool codeword_found;
    for (int curr_cdc_iter = 0; curr_cdc_iter < max_iters; curr_cdc_iter++) {
        ::cuda::gputimer t_spa_iteration("t_spa_iteration");
        codeword_found = this->spa_iteration();
        this->add_timer(t_spa_iteration);

        if (codeword_found)
            break;
    }

    // Copy the received codeword from the GPU.
    ::cuda::gputimer t_copy_codeword("t_copy_codeword_d_to_h");
    received_word = (libbase::vector<GF_q>)this->device_received_word;
    this->add_timer(t_copy_codeword);
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
