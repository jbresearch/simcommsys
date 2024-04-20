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
#define ROUND_UP_DIV(X, Y) (((X) + (Y)-1) / (Y))

// Declarations
// ----------------------------------------------------------

/*! \brief Compute LUT for multiplication in GF_q
 */
template <class GF_q>
__global__ void compute_perms(::cuda::matrix_reference<int> perms,
                              int num_of_elements);

/*! \brief Perform clipping of zero values to almost-zero values on device.
 */
template <class real>
__device__
void perform_clipping(real& num, int& clipping_method, real& almost_zero);

/*! \brief Kernel to apply clipping to and normalize probability distributions
 * in a matrix.
 *
 * This kernel operates on a matrix where each row is a probability
 * distribution. For each row, the kernel
 * - Clips values close to 0 (we decide on values to be clipped based on
 * \p clipping_method), and
 * - Normalizes all values in a probability distribution so that sum of the
 * distribution is 1.0, even after clipping.
 *
 * \param probs Matrix where each row is a probability distribution.
 *
 * \param clipping_method The clipping method used.
 *
 * \param almost_zero Value which clipped values are set to.
 */
template <class GF_q, class real>
__global__ void
clip_and_normalize_probs_kern(::cuda::matrix_reference<real> probs,
                              int clipping_method,
                              real almost_zero);

/*! \brief Initializes device arrays for SPA, particularly \p device_qmn_conv
 * and \p device_r_mxn.
 *
 * \p device_qmn_conv is initialized using the probabilities in
 * \p device_receieved_probs (permuted to account for multiplication by values
 * in the parity check matrix).
 *
 * \p device_r_mxn is zero initialized. TODO: Use cudaMemset to zero initialize
 * instead, also maybe we don't even need to zero initialize
 *
 * \param device_received_probs Prior probability distributions over GF(q) for
 * each symbol n
 *
 * \param device_perms Look up table for Galois field multiplication in GF_q
 *
 * \param device_qmn_row_indices m x n matrix containing indices of rows of
 * src/dst that contain distributions corresponding to (m, n).
 *
 * \param device_r_mxn Matrix where each row will hold the computed r_mxn
 * message for a particular (m, n). The mapping between (m, n) and rows is given
 * by \p device_qmn_row_indices.
 *
 * \param device_qmn_conv Matrix where each row holds the Hadamard transform for
 * q_mxn messages used to compute the "r_mxn"s for a particular (m, n). The
 * mapping between (m, n) and rows is given by \p device_qmn_row_indices.
 *
 * \param device_pchk_row_non_zeros m-size vector containing number of non-zeros
 * per row of the parity check matrix.
 *
 * \param device_pchk_row_non_zeros_pos m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the index positions (0-indexed) of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param device_pchk_row_non_zeros_val m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the values in GF_q of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 */
template <class GF_q, class real>
__global__ void
spa_init_kern(::cuda::matrix_reference<real> device_received_probs,
              ::cuda::matrix_reference<int> device_perms,
              ::cuda::matrix_reference<int> device_qmn_row_indices,
              ::cuda::matrix_reference<real> device_r_mxn,
              ::cuda::matrix_reference<real> device_qmn_conv,
              ::cuda::vector_reference<int> device_pchk_row_non_zeros,
              ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
              ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val);

/*! \brief Wrapper for clip_and_normalize_probs_kern().
 *
 * Takes care of the kernel call, including passing the size of dynamically
 * allocated shared memory used by the kernel.
 *
 * \param probs Matrix where each row is a probability distribution.
 *
 * \param clipping_method The clipping method used.
 *
 * \param almost_zero Value which clipped values are set to.
 */
template <class GF_q, class real>
void clip_and_normalize_probs(::cuda::matrix_reference<real> probs,
                              int clipping_method,
                              real almost_zero);

/*! \brief Performs a single "butterfly" pass of the Hadamard-Walsh transform
 * over a number of probability distributions.
 *
 * The pass uses the butterfly property to permute distributions in \p src into
 * \p dst in a cache-efficient way.
 *
 * \param src n x |GF_q| matrix containing n distributions over
 * GF_q to which the transform will be applied.
 *
 * \param dst n x |GF_q| matrix that result of transform on src
 * will be stored in.
 *
 * \param tanner_edges Number of distributions that transform will be applied
 * to. The name of the arg comes from the use of this function in SPA, where
 * number of distributions is equal to the edges in the Tanner graph.
 *
 * \param h Indicates the distance used in the "butterfly" pass (elements this
 * distance apart within a single row of src are combined). Starts out at
 * |GF_q|/2 and is divided by 2 at each pass.
 */
template <class GF_q, class real>
__global__ void hadamard_transform_pass_kern(::cuda::matrix_reference<real> src,
                                             ::cuda::matrix_reference<real> dst,
                                             int tanner_edges,
                                             int h);

/*! \brief Perform a permutation of \p src into \p dst.
 *
 * The permutation is such that for a check m and a symbol n which participates
 * in the check:
 *
 * dst(device_qmn_row_indices(m,n), h_m_n*e) =
 * src(device_qmn_row_indices(m,n),e)
 *
 * for every element e in GF_q, and h_m_n is a parity check matrix
 * element corresponding to (m, n).
 *
 * So effectively we are transforming a distribution over a random variable e in
 * GF_q to a distribution over h_m_n*e
 *
 * \param device_perms Look up table for Galois field multiplication in GF_q
 *
 * \param device_qmn_row_indices m x n matrix containing indices of rows of
 * src/dst that contain distributions corresponding to (m, n).
 *
 * \param src Matrix where each row is a "probability" distribution over GF_q
 * corresponding to some (m, n) pair.
 *
 * \param dst Destination matrix containing permuted distributions from src
 * (result of this kernel).
 *
 * \param device_pchk_row_non_zeros m-size vector containing number of non-zeros
 * per row of the parity check matrix.
 *
 * \param device_pchk_row_non_zeros_pos m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the index positions (0-indexed) of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param device_pchk_row_non_zeros_val m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the values in GF_q of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 */
template <class GF_q, class real>
__global__ void multiply_h_m_n_kern(
    ::cuda::matrix_reference<int> device_perms,
    ::cuda::matrix_reference<int> device_qmn_row_indices,
    ::cuda::matrix_reference<real> src,
    ::cuda::matrix_reference<real> dst,
    ::cuda::vector_reference<int> device_pchk_row_non_zeros,
    ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
    ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val);

/*! \brief Perform a permutation of \p src into \p dst.
 *
 * The permutation is such that for a check m and a symbol n which participates
 * in the check:
 *
 * dst(device_qmn_row_indices(m,n),e) =
 * src(device_qmn_row_indices(m,n),h_m_n*e)
 *
 * for every element e in GF_q, and h_m_n is a parity check matrix
 * element corresponding to (m, n).
 *
 * So effectively we are transforming a distribution over a random variable
 * h_m_n*e in GF_q to a distribution over e
 *
 * \param device_perms Look up table for Galois field multiplication in GF_q
 *
 * \param device_qmn_row_indices m x n matrix containing indices of rows of
 * src/dst that contain distributions corresponding to (m, n).
 *
 * \param src Matrix where each row is a "probability" distribution over GF_q
 * corresponding to some (m, n) pair.
 *
 * \param dst Destination matrix containing permuted distributions from src
 * (result of this kernel).
 *
 * \param device_pchk_row_non_zeros m-size vector containing number of non-zeros
 * per row of the parity check matrix.
 *
 * \param device_pchk_row_non_zeros_pos m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the index positions (0-indexed) of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param device_pchk_row_non_zeros_val m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the values in GF_q of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 */
template <class GF_q, class real>
__global__ void
divide_h_m_n_kern(::cuda::matrix_reference<int> device_perms,
                  ::cuda::matrix_reference<int> device_qmn_row_indices,
                  ::cuda::matrix_reference<real> src,
                  ::cuda::matrix_reference<real> dst,
                  ::cuda::vector_reference<int> device_pchk_row_non_zeros,
                  ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
                  ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val);

/*! \brief Compute the Hadamard transform of each "probability" distribution in
 * \p src, and store result in \p dst.
 *
 * \param src Matrix where each row is a "probability" distribution over GF_q
 * \param dst Matrix of same dimensions as src used to store result of Hadamard
 * transforms.
 */
template <class GF_q, class real>
void hadamard_transform(::cuda::matrix_reference<real> src,
                        ::cuda::matrix_reference<real> dst);

/*! \brief Compute r_mxn messages from \p device_qmn_conv. Results are stored in
 * \p device_r_mxn.
 *
 * Note that if \p device_qmn_conv stores the Hadamard transform of the actual
 * "q_mxn"s, as in our impl., this kernel is not enough to compute the r_mn
 * messages, but we need to apply the Hadamard transform on its results.
 *
 * \param device_qmn_row_indices m x n matrix containing indices of rows of
 * src/dst that contain distributions corresponding to (m, n).
 *
 * \param device_r_mxn Matrix where each row will hold the computed r_mxn
 * message for a particular (m, n). The mapping between (m, n) and rows is given
 * by \p device_qmn_row_indices.
 *
 * \param device_qmn_conv Matrix where each row holds the Hadamard transform of
 * q_mxn messages used to compute the "r_mxn"s for a particular (m, n). The
 * mapping between (m, n) and rows is given by \p device_qmn_row_indices.
 *
 * \param device_pchk_row_non_zeros m-size vector containing number of non-zeros
 * per row of the parity check matrix.
 *
 * \param device_pchk_row_non_zeros_pos m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the index positions (0-indexed) of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 */
template <class GF_q, class real>
__global__ void
compute_r_mn_kern(::cuda::matrix_reference<int> device_qmn_row_indices,
                  ::cuda::matrix_reference<real> device_r_mxn,
                  ::cuda::matrix_reference<real> device_qmn_conv,
                  ::cuda::vector_reference<int> device_pchk_row_non_zeros,
                  ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos);

/*! \brief Full computation of r_mxn messages in the context of our algorithm.
 * Results are stored in \p device_r_mxn.
 *
 * This function takes the following steps:
 * - Uses the compute_r_mn_kern() kernel to compute messages from
 * \p device_qmn_conv.
 *
 * - Applies the Hadamard transform to the result of the last
 * step
 *
 * - Applies the divide_h_m_n_kern to the result of the last step; this gives us
 * the actual r_mxn messages.
 *
 * - Applies clipping and normalization to the r_mxn messages computed in the
 * last step
 *
 * \param device_perms Look up table for Galois field multiplication in GF_q
 *
 * \param device_qmn_row_indices m x n matrix containing indices of rows of
 * src/dst that contain distributions corresponding to (m, n).
 *
 * \param device_r_mxn Matrix where each row will hold the computed r_mxn
 * message for a particular (m, n). The mapping between (m, n) and rows is given
 * by \p device_qmn_row_indices.
 *
 * \param device_qmn_conv Matrix where each row holds the Hadamard transform of
 * q_mxn messages used to compute the "r_mxn"s for a particular (m, n). The
 * mapping between (m, n) and rows is given by \p device_qmn_row_indices.
 *
 * \param device_pchk_row_non_zeros m-size vector containing number of non-zeros
 * per row of the parity check matrix.
 *
 * \param device_pchk_row_non_zeros_pos m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the index positions (0-indexed) of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param device_pchk_row_non_zeros_val m x max(device_pchk_row_non_zeros)
 * matrix where each row contains the values in GF_q of non-zero
 * values in the corresponding row of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param device_swap_buf Matrix of same size as \p device_r_mxn which is used
 * as a swap buffer when Hadamard transform/division by h_m_n values are being
 * computed.
 */
template <class GF_q, class real>
void compute_r_mn(::cuda::matrix<int>& device_perms,
                  ::cuda::matrix<int>& device_qmn_row_indices,
                  ::cuda::matrix<real>& device_r_mxn,
                  ::cuda::matrix<real>& device_qmn_conv,
                  ::cuda::vector<int>& device_pchk_row_non_zeros,
                  ::cuda::matrix<int>& device_pchk_row_non_zeros_pos,
                  ::cuda::matrix<GF_q>& device_pchk_row_non_zeros_val,
                  ::cuda::matrix<real>& device_swap_buf,
                  int clipping_method,
                  real almost_zero);

template <class GF_q, class real>
__global__ void
compute_q_mn_kern(::cuda::matrix_reference<real> device_received_probs,
                  ::cuda::matrix_reference<int> device_qmn_row_indices,
                  ::cuda::matrix_reference<real> device_r_mxn,
                  ::cuda::matrix_reference<real> device_qmn_conv,
                  ::cuda::vector_reference<int> device_pchk_col_non_zeros,
                  ::cuda::matrix_reference<int> device_pchk_col_non_zeros_pos);

template <class GF_q, class real>
void compute_q_mn(::cuda::matrix<real>& device_received_probs,
                  ::cuda::matrix<int>& device_perms,
                  ::cuda::matrix<int>& device_qmn_row_indices,
                  ::cuda::matrix<real>& device_r_mxn,
                  ::cuda::matrix<real>& device_qmn_conv,
                  ::cuda::vector<int>& device_pchk_row_non_zeros,
                  ::cuda::matrix<int>& device_pchk_row_non_zeros_pos,
                  ::cuda::matrix<GF_q>& device_pchk_row_non_zeros_val,
                  ::cuda::vector<int>& device_pchk_col_non_zeros,
                  ::cuda::matrix<int>& device_pchk_col_non_zeros_pos,
                  ::cuda::matrix<GF_q>& device_pchk_col_non_zeros_val,
                  ::cuda::matrix<real>& device_swap_buf,
                  int clipping_method,
                  real almost_zero);

template <class GF_q, class real>
__global__ void compute_probs_kern(
    ::cuda::matrix_reference<real> device_received_probs,
    ::cuda::matrix_reference<int> device_qmn_row_indices,
    ::cuda::matrix_reference<real> device_r_mxn,
    ::cuda::vector_reference<int> device_pchk_col_non_zeros,
    ::cuda::matrix_reference<int> device_pchk_col_non_zeros_pos,
    ::cuda::matrix_reference<GF_q> device_pchk_col_non_zeros_val);

/*! \brief Compute posterior probabilities from r_mn messages in \p device_r_mxn
 * . The results are stored in \p device_received_probs.
 *
 * For a particular symbol n, the posterior probability that symbol n has value
 * e is the product of
 *
 * device_r_mxn(device_qmn_row_indices(m, n), e)
 *
 * where m ranges over checks which symbol n participates in.
 *
 * \param device_receieved_probs n x |GF(q)| matrix where the posterior
 * probability distributions will be stored.
 *
 * \param device_qmn_row_indices m x n matrix containing indices of rows of
 * src/dst that contain distributions corresponding to (m, n).
 *
 * \param device_r_mxn Matrix where each row will hold the computed r_mxn
 * message for a particular (m, n). The mapping between (m, n) and rows is given
 * by \p device_qmn_row_indices.
 *
 * \param device_pchk_col_non_zeros n-size vector containing number of non-zeros
 * per column of the parity check matrix.
 *
 * \param device_pchk_col_non_zeros_pos n x max(device_pchk_col_non_zeros)
 * matrix where each row contains the index positions (0-indexed) of non-zero
 * values in a column of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param device_pchk_col_non_zeros_val n x max(device_pchk_col_non_zeros)
 * matrix where each row contains the values in GF_q of non-zero
 * values in a column of the parity check matrix. Extra slots at
 * the end of each row are padded with zeros.
 *
 * \param clipping_method The clipping method used to determine which almost
 * zero/zero values to clip.
 *
 * \param almost_zero Clipped values are set to this value.
 */
template <class GF_q, class real>
void compute_probs(::cuda::matrix<real>& device_received_probs,
                   ::cuda::matrix<int>& device_qmn_row_indices,
                   ::cuda::matrix<real>& device_r_mxn,
                   ::cuda::vector<int>& device_pchk_col_non_zeros,
                   ::cuda::matrix<int>& device_pchk_col_non_zeros_pos,
                   ::cuda::matrix<GF_q>& device_pchk_col_non_zeros_val,
                   int clipping_method,
                   real almost_zero);

// Definitions
// ----------------------------------------------------------

template <class GF_q>
__global__ void
compute_perms(::cuda::matrix_reference<int> perms)
{
    int num_of_elements = GF_q::elements();

    // use 1D index to take advantage of memory layout of perms (row-major)
    // Note that there is no need for bounds checking here since num_of_elements
    // is always a power of two. Hence granted blockDim is a power of two as
    // well, num_of_elements * num_of_elements is divided perfectly by blockDim.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int ix = min(i % num_of_elements, num_of_elements - 1);
    int iy = min(i / num_of_elements, num_of_elements - 1);

    perms(ix, iy) = GF_q(ix) * GF_q(iy);
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

    device_perms.init(num_of_elements, num_of_elements);

    int log_block_dim = 10;
    int block_dim = 1 << log_block_dim;
    int num_blocks = (num_of_elements * num_of_elements) >> log_block_dim;
    num_blocks = max(num_blocks, 1);

    // we can use shift for dividing since block size is a power of two.
    // Note that there is no need to account for division that rounds
    // towards zero since num_of_elements is always a power of two for GF_q.
    // Hence granted blockDim is a power of two as well, num_of_elements *
    // num_of_elements is divided perfectly by blockDim.
    compute_perms<GF_q><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int>(device_perms));
    cudaSafeCall(cudaGetLastError());

    // we first build qmn_row_indices on the host, then copy to device.
    // Easier since this operation is inherently serial (we have a counter
    // to keep track of current index) and also we need the tanner_edges var
    // computed during this process on host to allocate memory for qmn and
    // rmn matrices.
    matrixi_t qmn_row_indices(m, n);
    // fill with -1 initially (means bit n does not participate in check m)
    qmn_row_indices = -1;

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
    matrixi_t pchk_row_non_zeros_pos(m, max_pchk_row_non_zeros);
    libbase::matrix<GF_q> pchk_row_non_zeros_val(m, max_pchk_row_non_zeros);

    matrixi_t pchk_col_non_zeros_pos(n, max_pchk_col_non_zeros);
    libbase::matrix<GF_q> pchk_col_non_zeros_val(n, max_pchk_col_non_zeros);

    // counts the number of edges in the Tanner graph of the code.
    // Tells us what the size of device_rmxn and device_qmn_conv should
    // be.
    int tanner_edges = 0;

    // Populate qmn_row_indices
    // Also populate the rest of the parity check matrix repr. on the host.
    // Actual n value, since loop_n is just an index ranging over the number of
    // non-zero values in a row of pchk_matrix.
    int pos_n;
    GF_q val;
    for (int loop_m = 0; loop_m < m; loop_m++) {
        // non-zeros for this row of the parity check matrix
        non_zeros = pchk_row_non_zeros(loop_m);

        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            pos_n = non_zero_row_pos(loop_m)(loop_n) - 1; // we count from zero;
            val = pchk_matrix(loop_m, pos_n);

            // populate other pchk matrix fields on the host.
            pchk_row_non_zeros_pos(loop_m, loop_n) = pos_n;
            pchk_row_non_zeros_val(loop_m, loop_n) = val;

            // assign an index in device_q_mn_conv, device_r_mxn and so on
            // to a non-zero (m, n) element.
            qmn_row_indices(loop_m, pos_n) = tanner_edges;
            tanner_edges++;
        }
    }

    // Actual m value, since loop_m is just an index ranging over the number of
    // non-zero values in a column of pchk_matrix.
    int pos_m;
    for (int loop_n = 0; loop_n < n; loop_n++) {
        // non-zeros for this col of the parity check matrix
        non_zeros = pchk_col_non_zeros(loop_n);

        for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
            pos_m = non_zero_col_pos(loop_n)(loop_m) - 1; // we count from zero;
            val = pchk_matrix(pos_m, loop_n);

            // populate other pchk matrix fields on the host.
            pchk_col_non_zeros_pos(loop_n, loop_m) = pos_m;
            pchk_col_non_zeros_val(loop_n, loop_m) = val;
        }
    }

    device_qmn_row_indices.init(m, n);
    // Copy qmn_row_indices to device
    device_qmn_row_indices = qmn_row_indices;

    // Allocate memory on the device for representation of the parity check
    // matrix
    device_pchk_row_non_zeros.init(m);
    device_pchk_row_non_zeros_pos.init(m, max_pchk_row_non_zeros);
    device_pchk_row_non_zeros_val.init(m, max_pchk_row_non_zeros);

    device_pchk_col_non_zeros.init(n);
    device_pchk_col_non_zeros_pos.init(n, max_pchk_col_non_zeros);
    device_pchk_col_non_zeros_val.init(n, max_pchk_col_non_zeros);

    // Copy represenation of the parity check matrix to the device.
    device_pchk_row_non_zeros = pchk_row_non_zeros;
    device_pchk_row_non_zeros_pos = pchk_row_non_zeros_pos;
    device_pchk_row_non_zeros_val = pchk_row_non_zeros_val;

    device_pchk_col_non_zeros = pchk_col_non_zeros;
    device_pchk_col_non_zeros_pos = pchk_col_non_zeros_pos;
    device_pchk_col_non_zeros_val = pchk_col_non_zeros_val;

    // Allocate required memory for r_mxn, q_mxn and qmn_conv on device.
    device_r_mxn.init(tanner_edges, num_of_elements);
    device_qmn_conv.init(tanner_edges, num_of_elements);

    device_swap_buf.init(tanner_edges, num_of_elements);
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
        perform_clipping(tmp_prob, clipping_method, almost_zero);
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
clip_and_normalize_probs(::cuda::matrix_reference<real> probs,
                         int clipping_method,
                         real almost_zero)
{
    int n = probs.get_rows();

    dim3 block_dim(1024);
    dim3 num_blocks(ROUND_UP_DIV(n, (int)block_dim.x));

    clip_and_normalize_probs_kern<GF_q, real>
        <<<num_blocks, block_dim>>>(probs, clipping_method, almost_zero);
    cudaSafeCall(cudaGetLastError());
}

template <class GF_q, class real>
__global__ void
spa_init_kern(::cuda::matrix_reference<real> device_received_probs,
              ::cuda::matrix_reference<int> device_perms,
              ::cuda::matrix_reference<int> device_qmn_row_indices,
              ::cuda::matrix_reference<real> device_r_mxn,
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
    loop_e = min(loop_e, num_of_elements - 1);

    int non_zeros = device_pchk_row_non_zeros(loop_m);

    int qmn_row_idx;
    int pos;
    GF_q h_m_n;
    // NOTE: loop_n iterates over number of symbols that participate in mth
    // check of a codeword. E.g. if check involves {x_1, x_4, x_6}, loop_n
    // ranges over [0, 1, 2]
    for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
        // NOTE: pos is the actual index of the nth symbol participating in the
        // mth check in the codeword. E.g. if check involves {x_1, x_4, x_6}
        // and loop_n = 1, pos = 4 (-1 since we count from 0)
        pos = device_pchk_row_non_zeros_pos(loop_m, loop_n);
        // NOTE: Find corresponding value in the parity check matrix.
        // We use loop_m because this is the check index, and pos because
        // this is the actual index of the nth symbol participating in the mth
        // check (non_zeros variable does not count symbols that don't
        // participate in the mth check).
        h_m_n = device_pchk_row_non_zeros_val(loop_m, loop_n);

        // get index into device_qmn_conv and device_r_mxn
        qmn_row_idx = device_qmn_row_indices(loop_m, pos);

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

    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(dim_n, (int)block_dim.x),
                      ROUND_UP_DIV(num_of_elements, (int)block_dim.y));
    spa_init_kern<GF_q, real>
        <<<num_blocks, block_dim>>>(this->device_received_probs,
                                    this->device_perms,
                                    this->device_qmn_row_indices,
                                    this->device_r_mxn,
                                    this->device_qmn_conv,
                                    this->device_pchk_row_non_zeros,
                                    this->device_pchk_row_non_zeros_pos,
                                    this->device_pchk_row_non_zeros_val);
    cudaSafeCall(cudaGetLastError());

    // apply the FFT again to get the proper values
    // Here we use matrix references for cheap swapping. The result of the
    // Hadamard transform will always be in src.
    ::cuda::matrix_reference<real> src(device_qmn_conv);
    ::cuda::matrix_reference<real> dst(device_swap_buf);
    hadamard_transform<GF_q, real>(src, dst);

    // Result of the Hadamard transform is always stored in src, copy to
    // device_qmn_conv in case src is the swap buffer.
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
hadamard_transform_pass_kern(::cuda::matrix_reference<real> src,
                             ::cuda::matrix_reference<real> dst,
                             int tanner_edges,
                             int h)
{
    // this is just a generic index that ranges over [0, tanner_edges)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = min(i, tanner_edges - 1);

    // find loop_e
    int loop_e = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

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
    ::cuda::matrix_reference<int> device_perms,
    ::cuda::matrix_reference<int> device_qmn_row_indices,
    ::cuda::matrix_reference<real> src,
    ::cuda::matrix_reference<real> dst,
    ::cuda::vector_reference<int> device_pchk_row_non_zeros,
    ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
    ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    // find loop_m
    int loop_m = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int m = device_pchk_row_non_zeros.size();
    loop_m = min(loop_m, m - 1);

    // find loop_e
    int loop_e = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

    int non_zeros = device_pchk_row_non_zeros(loop_m);
    // actual value of n (loop_n ranges over the number of symbols in check m)
    int pos_n;
    // hold value of pchk matrix at (m, n)
    int h_m_n;
    for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
        pos_n = device_pchk_row_non_zeros_pos(loop_m, loop_n);
        h_m_n = device_pchk_row_non_zeros_val(loop_m, loop_n);
        // perform the permutation
        dst(device_qmn_row_indices(loop_m, pos_n),
            device_perms(h_m_n, loop_e)) =
            src(device_qmn_row_indices(loop_m, pos_n), loop_e);
    }
}

template <class GF_q, class real>
__global__ void
divide_h_m_n_kern(::cuda::matrix_reference<int> device_perms,
                  ::cuda::matrix_reference<int> device_qmn_row_indices,
                  ::cuda::matrix_reference<real> src,
                  ::cuda::matrix_reference<real> dst,
                  ::cuda::vector_reference<int> device_pchk_row_non_zeros,
                  ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos,
                  ::cuda::matrix_reference<GF_q> device_pchk_row_non_zeros_val)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    // find loop_m
    int loop_m = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int m = device_pchk_row_non_zeros.size();
    loop_m = min(loop_m, m - 1);

    // find loop_e
    int loop_e = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

    int non_zeros = device_pchk_row_non_zeros(loop_m);
    // actual value of n (loop_n ranges over the number of symbols in check m)
    int pos_n;
    // hold value of pchk matrix at (m, n)
    int h_m_n;
    for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
        pos_n = device_pchk_row_non_zeros_pos(loop_m, loop_n);
        h_m_n = device_pchk_row_non_zeros_val(loop_m, loop_n);
        // perform the permutation
        dst(device_qmn_row_indices(loop_m, pos_n), loop_e) = src(
            device_qmn_row_indices(loop_m, pos_n), device_perms(h_m_n, loop_e));
    }
}

template <class GF_q, class real>
inline void
hadamard_transform(::cuda::matrix_reference<real> src,
                   ::cuda::matrix_reference<real> dst)
{
    // src and dst need to have the same dimensions
    cuda_assert(src.get_cols() == dst.get_cols() &&
                src.get_rows() == dst.get_rows());

    int num_of_elements = GF_q::elements();
    int tanner_edges = src.get_rows();

    dim3 block_dim = dim3(32, 32);
    // use division which truncates upwards.
    dim3 num_blocks = dim3(ROUND_UP_DIV(tanner_edges, (int)block_dim.x),
                           ROUND_UP_DIV(num_of_elements, (int)block_dim.y));

    int h;
    for (h = 1; h < num_of_elements; h <<= 1) {
        hadamard_transform_pass_kern<GF_q, real>
            <<<num_blocks, block_dim>>>(src, dst, tanner_edges, h);
        cudaSafeCall(cudaGetLastError());

        std::swap(src, dst);
    }
}

template <class GF_q, class real>
__global__ void
compute_r_mn_kern(::cuda::matrix_reference<int> device_qmn_row_indices,
                  ::cuda::matrix_reference<real> device_r_mxn,
                  ::cuda::matrix_reference<real> device_qmn_conv,
                  ::cuda::vector_reference<int> device_pchk_row_non_zeros,
                  ::cuda::matrix_reference<int> device_pchk_row_non_zeros_pos)
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
    loop_e = min(loop_e, num_of_elements - 1);

    int non_zeros = device_pchk_row_non_zeros(loop_m);
    // actual value of n (loop_n ranges over the number of symbols in check m)
    int pos_n;
    // if message is being computed to send over edge from m to n, then this
    // ranges over all other symbols that participate in check m but are not n,
    // i.e. all symbols included in the message.
    int pos_n_dash;
    // Holds the actual message computed
    real q_nm_conv_prod;
    for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
        q_nm_conv_prod = 1.0;
        pos_n = device_pchk_row_non_zeros_pos(loop_m, loop_n);

        for (int loop_n_dash = 0; loop_n_dash < non_zeros; loop_n_dash++) {
            pos_n_dash = device_pchk_row_non_zeros_pos(loop_m, loop_n_dash);

            q_nm_conv_prod *=
                // we multiply by the check pos_n_dash != pos_n to ensure that
                // symbol n itself is not included in the message
                (pos_n_dash != pos_n) *
                device_qmn_conv(device_qmn_row_indices(loop_m, pos_n_dash),
                                loop_e);
        }
        // Loop above has potential divergence as different m have different
        // degrees in general. We want to convergence again here so most iters
        // are in sync.
        // TODO: Test impact of this.
        __syncthreads();
        // coalesced memory access due to syncthreads above
        // We store in r_mxn but this is not the final result.
        device_r_mxn(device_qmn_row_indices(loop_m, pos_n), loop_e) =
            q_nm_conv_prod;
    }
}

template <class GF_q, class real>
void
compute_r_mn(::cuda::matrix<int>& device_perms,
             ::cuda::matrix<int>& device_qmn_row_indices,
             ::cuda::matrix<real>& device_r_mxn,
             ::cuda::matrix<real>& device_qmn_conv,
             ::cuda::vector<int>& device_pchk_row_non_zeros,
             ::cuda::matrix<int>& device_pchk_row_non_zeros_pos,
             ::cuda::matrix<GF_q>& device_pchk_row_non_zeros_val,
             ::cuda::matrix<real>& device_swap_buf,
             int clipping_method,
             real almost_zero)
{
    dim3 block_dim, num_blocks;

    int m = device_pchk_row_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(m, (int)block_dim.x),
                      ROUND_UP_DIV(num_of_elements, (int)block_dim.y));
    compute_r_mn_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int>(device_qmn_row_indices),
        ::cuda::matrix_reference<real>(device_r_mxn),
        ::cuda::matrix_reference<real>(device_qmn_conv),
        ::cuda::vector_reference<int>(device_pchk_row_non_zeros),
        ::cuda::matrix_reference<int>(device_pchk_row_non_zeros_pos));
    cudaSafeCall(cudaGetLastError());

    // apply the FFT again to get the proper values
    // Here we use matrix references for cheap swapping. The result of the
    // Hadamard transform will always be in src.
    ::cuda::matrix_reference<real> src(device_r_mxn);
    ::cuda::matrix_reference<real> dst(device_swap_buf);
    hadamard_transform<GF_q, real>(src, dst);

    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(m, (int)block_dim.x),
                      ROUND_UP_DIV(num_of_elements, (int)block_dim.y));

    // Permute the distributions in src (transformed by the Hadamard transform)
    // into dst
    divide_h_m_n_kern<<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int>(device_perms),
        ::cuda::matrix_reference<int>(device_qmn_row_indices),
        src,
        dst,
        ::cuda::vector_reference<int>(device_pchk_row_non_zeros),
        ::cuda::matrix_reference<int>(device_pchk_row_non_zeros_pos),
        ::cuda::matrix_reference<GF_q>(device_pchk_row_non_zeros_val));
    cudaSafeCall(cudaGetLastError());

    // dst could be device_r_mxn or device_swap_buf depending on whether no. of
    // passes in Hadamard transform is even or odd. We copy back to device_r_mxn
    // to make sure the result is in the right array.
    device_r_mxn = dst;

    // Apply clipping + normalization to the computed r_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real>(device_r_mxn),
        clipping_method,
        almost_zero);
}

template <class GF_q, class real>
__global__ void
compute_q_mn_kern(::cuda::matrix_reference<real> device_received_probs,
                  ::cuda::matrix_reference<int> device_qmn_row_indices,
                  ::cuda::matrix_reference<real> device_r_mxn,
                  ::cuda::matrix_reference<real> device_qmn_conv,
                  ::cuda::vector_reference<int> device_pchk_col_non_zeros,
                  ::cuda::matrix_reference<int> device_pchk_col_non_zeros_pos)
{
    // find loop_n
    int loop_n = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    loop_n = min(loop_n, n - 1);

    // find loop_e
    int loop_e = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

    // Current probability that received symbol n has value e.
    real recvd_prob = device_received_probs(loop_n, loop_e);

    int non_zeros = device_pchk_col_non_zeros(loop_n);
    // actual value of m (loop_m ranges over the number of symbols in check m)
    int pos_m;
    // if message is being computed to send over edge from n to m, then this
    // ranges over all checks which n participates in.
    int pos_m_dash;
    // Holds the actual message computed
    real q_nm;
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        q_nm = recvd_prob;
        pos_m = device_pchk_col_non_zeros_pos(loop_m, loop_n);

        for (int loop_m_dash = 0; loop_m_dash < non_zeros; loop_m_dash++) {
            pos_m_dash = device_pchk_col_non_zeros_pos(loop_m_dash, loop_n);

            q_nm *=
                // we multiply by the check pos_m_dash != pos_m to ensure that
                // check m itself is not included in the message
                (pos_m_dash != pos_m) *
                device_r_mxn(device_qmn_row_indices(pos_m_dash, loop_n),
                             loop_e);
        }
        // Loop above has potential divergence as different m have different
        // degrees in general. We want to convergence again here so most iters
        // are in sync.
        // TODO: Test impact of this.
        __syncthreads();

        // Uncoalesced memory access.
        device_qmn_conv(device_qmn_row_indices(pos_m, loop_n), loop_e) = q_nm;
    }
}

template <class GF_q, class real>
void
compute_q_mn(::cuda::matrix<real>& device_received_probs,
             ::cuda::matrix<int>& device_perms,
             ::cuda::matrix<int>& device_qmn_row_indices,
             ::cuda::matrix<real>& device_r_mxn,
             ::cuda::matrix<real>& device_qmn_conv,
             ::cuda::vector<int>& device_pchk_row_non_zeros,
             ::cuda::matrix<int>& device_pchk_row_non_zeros_pos,
             ::cuda::matrix<GF_q>& device_pchk_row_non_zeros_val,
             ::cuda::vector<int>& device_pchk_col_non_zeros,
             ::cuda::matrix<int>& device_pchk_col_non_zeros_pos,
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
    num_blocks = dim3(ROUND_UP_DIV(n, (int)block_dim.x),
                      ROUND_UP_DIV(num_of_elements, (int)block_dim.y));
    compute_q_mn_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<real>(device_received_probs),
        ::cuda::matrix_reference<int>(device_qmn_row_indices),
        ::cuda::matrix_reference<real>(device_r_mxn),
        ::cuda::matrix_reference<real>(device_qmn_conv),
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros),
        ::cuda::matrix_reference<int>(device_pchk_col_non_zeros_pos));
    cudaSafeCall(cudaGetLastError());

    // Apply clipping + normalization to the computed q_mn values.
    clip_and_normalize_probs<GF_q, real>(
        ::cuda::matrix_reference<real>(device_qmn_conv),
        clipping_method,
        almost_zero);

    // Here we use matrix references for cheap swapping.
    ::cuda::matrix_reference<real> src(device_r_mxn);
    ::cuda::matrix_reference<real> dst(device_swap_buf);

    int m = device_pchk_row_non_zeros.size();
    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(m, (int)block_dim.x),
                      ROUND_UP_DIV(num_of_elements, (int)block_dim.y));

    // Permute the distributions in src into dst
    multiply_h_m_n_kern<<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<int>(device_perms),
        ::cuda::matrix_reference<int>(device_qmn_row_indices),
        src,
        dst,
        ::cuda::vector_reference<int>(device_pchk_row_non_zeros),
        ::cuda::matrix_reference<int>(device_pchk_row_non_zeros_pos),
        ::cuda::matrix_reference<GF_q>(device_pchk_row_non_zeros_val));
    cudaSafeCall(cudaGetLastError());
    std::swap(src, dst);

    // Compute Hadamard transform on the result.
    hadamard_transform<GF_q, real>(src, dst);

    // Result of the Hadamard transform is always stored in src, copy to
    // device_qmn_conv in case src is the swap buffer.
    device_qmn_conv = src;
}

template <class GF_q, class real>
__global__ void
compute_probs_kern(::cuda::matrix_reference<real> device_received_probs,
                   ::cuda::matrix_reference<int> device_qmn_row_indices,
                   ::cuda::matrix_reference<real> device_r_mxn,
                   ::cuda::vector_reference<int> device_pchk_col_non_zeros,
                   ::cuda::matrix_reference<int> device_pchk_col_non_zeros_pos,
                   ::cuda::matrix_reference<GF_q> device_pchk_col_non_zeros_val)
{

    // find loop_n
    int loop_n = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds checking
    int n = device_pchk_col_non_zeros.size();
    loop_n = min(loop_n, n - 1);

    // find loop_e
    int loop_e = blockIdx.y * blockDim.y + threadIdx.y;
    // bounds checking
    int num_of_elements = GF_q::elements();
    loop_e = min(loop_e, num_of_elements - 1);

    int non_zeros = device_pchk_col_non_zeros(loop_n);
    // actual value of m (loop_m ranges over the number of symbols in check m)
    int pos_m;
    // Holds the prob computed
    real prob;
    for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
        prob = 1.0;
        pos_m = device_pchk_col_non_zeros_pos(loop_m, loop_n);

        prob *= device_r_mxn(device_qmn_row_indices(pos_m, loop_n), loop_e);
    }

    device_received_probs(loop_n, loop_e) = prob;
}

template <class GF_q, class real>
void
compute_probs(::cuda::matrix<real>& device_received_probs,
              ::cuda::matrix<int>& device_qmn_row_indices,
              ::cuda::matrix<real>& device_r_mxn,
              ::cuda::vector<int>& device_pchk_col_non_zeros,
              ::cuda::matrix<int>& device_pchk_col_non_zeros_pos,
              ::cuda::matrix<GF_q>& device_pchk_col_non_zeros_val,
              int clipping_method,
              real almost_zero)
{
    dim3 block_dim, num_blocks;

    int n = device_pchk_col_non_zeros.size();
    int num_of_elements = GF_q::elements();
    block_dim = dim3(32, 32);
    // use division which truncates upwards.
    num_blocks = dim3(ROUND_UP_DIV(n, (int)block_dim.x),
                      ROUND_UP_DIV(num_of_elements, (int)block_dim.y));
    compute_probs_kern<GF_q, real><<<num_blocks, block_dim>>>(
        ::cuda::matrix_reference<real>(device_received_probs),
        ::cuda::matrix_reference<int>(device_qmn_row_indices),
        ::cuda::matrix_reference<real>(device_r_mxn),
        ::cuda::vector_reference<int>(device_pchk_col_non_zeros),
        ::cuda::matrix_reference<int>(device_pchk_col_non_zeros_pos),
        ::cuda::matrix_reference<GF_q>(device_pchk_col_non_zeros_val));
    cudaSafeCall(cudaGetLastError());

    // Normalize the computed probabilities.
    clip_and_normalize_probs<GF_q, real>(
        device_received_probs, clipping_method, almost_zero);
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
    // N(m)\n}q_mxn(x_{n') ) Essentially, what we are doing is the following:
    // Assume x_n=0
    // we need to sum over all possibilities that such that the parity check is
    // satisfied, ie =0 if the parity check is satisfied the conditional
    // probability is 1 and 0 otherwise so we are simply adding up the products
    // for which the parity check is satisfied.

    compute_r_mn(this->device_perms,
                 this->device_qmn_row_indices,
                 this->device_r_mxn,
                 this->device_qmn_conv,
                 this->device_pchk_row_non_zeros,
                 this->device_pchk_row_non_zeros_pos,
                 this->device_pchk_row_non_zeros_val,
                 this->device_swap_buf,
                 this->clipping_method,
                 this->almostzero);

    // loop over all the symbol nodes - the vertical step

    compute_q_mn(this->device_received_probs,
                 this->device_perms,
                 this->device_qmn_row_indices,
                 this->device_r_mxn,
                 this->device_qmn_conv,
                 this->device_pchk_row_non_zeros,
                 this->device_pchk_row_non_zeros_pos,
                 this->device_pchk_row_non_zeros_val,
                 this->device_pchk_col_non_zeros,
                 this->device_pchk_col_non_zeros_pos,
                 this->device_pchk_col_non_zeros_val,
                 this->device_swap_buf,
                 this->clipping_method,
                 this->almostzero);

    // compute the new probabilities for all symbols given the information in
    // this iteration. This will be used in a tentative decoding to see whether
    // we have found a codeword
    compute_probs(this->device_received_probs,
                  this->device_qmn_row_indices,
                  this->device_r_mxn,
                  this->device_pchk_col_non_zeros,
                  this->device_pchk_col_non_zeros_pos,
                  this->device_pchk_col_non_zeros_val,
                  this->clipping_method,
                  this->almostzero);

    // Copy received probabilities from device to host.

    // ensure ro has the right size
    ro.init(this->device_received_probs.get_rows());

    for (int n = 0; n < ro.size(); n++) {
        // allocate memory on host for probability distribution of symbol n
        ro(n).init(this->device_received_probs.get_cols());
        ro(n) =
            (libbase::vector<real>)this->device_received_probs.extract_row(n);
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
