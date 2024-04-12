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

#ifndef SUM_PROD_ALG_GDL_H_
#define SUM_PROD_ALG_GDL_H_

#include "../sum_prod_alg_inf.h"
#include "cuda/matrix.h"
#include "cuda/vector.h"
#include "matrix.h"
#include "vector.h"
#include <limits>
#include <string>

namespace libcomm
{

template <class GF_q>
__global__ void
compute_perms(::cuda::matrix_reference<int> perms, int num_of_elements)
{
    // use 1D index to take advantage of memory layout of perms (row-major)
    // Note that there is no need for bounds checking here since num_of_elements
    // is always a power of two. Hence granted blockDim is a power of two as
    // well, num_of_elements * num_of_elements is divided perfectly by blockDim.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int ix = i % num_of_elements;
    int iy = i / num_of_elements;

    perms(ix, iy) = GF_q(ix) * GF_q(iy);
}

template <class GF_q, class real = double>
class sum_prod_alg_gdl_cuda : public sum_prod_alg_inf<GF_q, real>
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<real> array1d_t;
    typedef libbase::vector<int> array1i_t;
    typedef libbase::vector<array1i_t> array1vi_t;
    typedef libbase::vector<array1d_t> array1vd_t;
    typedef libbase::matrix<int> matrixi_t;
    typedef ::cuda::vector<int> cuda_array1i_t;
    typedef ::cuda::matrix<int> cuda_matrixi_t;
    typedef ::cuda::matrix<real> cuda_matrixd_t;
    // @}

    /*! \brief constructor
     * This constructor calls the parent class but then
     * also creates a multiplication look-up table for the relevant
     * finite field. This is needed as multiplication in GF_q is fairly
     * expensive at the moment. Ideally, this look-up table should be moved
     * to the finite field implementation
     *
     */
    sum_prod_alg_gdl_cuda(int n,
                          int m,
                          const array1vi_t& non_zero_col_pos,
                          const array1vi_t& non_zero_row_pos,
                          const libbase::matrix<GF_q>& pchk_matrix)
    {
        int num_of_elements = GF_q::elements();

        device_perms.init(num_of_elements, num_of_elements);

        int log_block_dim = 5;
        int block_dim = 1 << log_block_dim;
        // we can use shift for dividing since block size is a power of two.
        // Note that there is no need to account for division that rounds
        // towards zero since num_of_elements is always a power of two for GF_q.
        // Hence granted blockDim is a power of two as well, num_of_elements *
        // num_of_elements is divided perfectly by blockDim.
        compute_perms<GF_q>
            <<<block_dim,
               ((num_of_elements * num_of_elements) >> log_block_dim)>>>(
                ::cuda::matrix_reference<int>(device_perms), num_of_elements);

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
            max_pchk_row_non_zeros =
                std::max(max_pchk_row_non_zeros, non_zeros);
        }

        // Find the maximum number of non zero elements in a col of the parity
        // check matrix.
        // Also populate pchk_col_non_zeros.
        max_pchk_col_non_zeros = std::numeric_limits<int>::min();
        for (int loop_n = 0; loop_n < n; loop_n++) {
            non_zeros = non_zero_col_pos(loop_n).size();

            pchk_col_non_zeros(loop_n) = non_zeros;
            max_pchk_col_non_zeros =
                std::max(max_pchk_col_non_zeros, non_zeros);
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
        int pos = 0;
        GF_q val;
        for (int loop_m = 0; loop_m < m; loop_m++) {
            // non-zeros for this row of the parity check matrix
            non_zeros = pchk_row_non_zeros(loop_m);

            for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
                pos =
                    non_zero_row_pos(loop_m)(loop_n) - 1; // we count from zero;
                val = pchk_matrix(loop_m, pos);

                // populate other pchk matrix fields on the host.
                pchk_row_non_zeros_pos(loop_m, loop_n) = pos;
                pchk_row_non_zeros_val(loop_m, loop_n) = val;

                // assign an index in device_q_mxn, device_r_mxn and so on to a
                // non-zero (m, n) element.
                qmn_row_indices(loop_m, pos) = tanner_edges;
                tanner_edges++;
            }
        }

        for (int loop_n = 0; loop_n < n; loop_n++) {
            // non-zeros for this col of the parity check matrix
            non_zeros = pchk_col_non_zeros(loop_n);

            for (int loop_m = 0; loop_m < non_zeros; loop_m++) {
                pos =
                    non_zero_col_pos(loop_n)(loop_m) - 1; // we count from zero;
                val = pchk_matrix(loop_n, pos);

                // populate other pchk matrix fields on the host.
                pchk_col_non_zeros_pos(loop_n, loop_m) = pos;
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
        device_q_mxn.init(tanner_edges, num_of_elements);
        device_qmn_conv.init(tanner_edges, num_of_elements);

        device_swap_buf.init(tanner_edges, num_of_elements);
    }
    virtual ~sum_prod_alg_gdl_cuda()
    {
        // nothing to do
    }

    // Overriden methods from sum_prod_alg_inf.
    void spa_init(const array1vd_t& ptable) override;
    void spa_iteration(array1vd_t& ro) override;
    std::string spa_type() override { return "gdl_cuda"; }

private:
    /*! \brief this holds a look-up table of the finite field multiplication
     */
    cuda_matrixi_t device_perms;

    cuda_matrixd_t device_received_probs;

    /*! Matrix of indices that tell us row of device_qmn_conv that
     * contains prob distribution qmn for a particular (m, n).
     * Also works for device_r_mxn since prob. distr. qmn and r_mxn have the
     * same size (size of GF(q) as there is one prob. for each element of GF(q))
     *
     * If bit n does not participate in check m, i.e. h_mn = 0, then the
     * corresponding element of this matrix is -1 by convention.
     */
    cuda_matrixi_t device_qmn_row_indices;

    /*! Each row of this matrix is a probability distribution r_mxn.
     * There is a row for each combination (m, n) such that bit n participates
     * in check m. The mapping between (m, n) and the rows is given by
     * device_qmn_row_indices
     */
    cuda_matrixd_t device_r_mxn;
    /*! Each row of this matrix is a probability distribution q_mn.
     * There is a row for each combination (m, n) such that bit n participates
     * in check m. The mapping between (m, n) and the rows is given by
     * device_qmn_row_indices
     */
    cuda_matrixd_t device_q_mxn;
    cuda_matrixd_t device_qmn_conv;
    /*! This is a swap buffer used for computing the Hadamard
     * transform/permutations on device_r_mxn or device_qmn_conv.
     *
     * It has the same dimensions as the latter fields.
     */
    cuda_matrixd_t device_swap_buf;

    /*! \name These fields are the representation of the parity check matrix in
     * device memory.
     */
    /*! \brief Maximum number of non-zero elements in a row of the parity check
     * matrix.
     */
    int max_pchk_row_non_zeros;
    /*! \brief Array containing number of non-zero elements in each row of
     * parity matrix h_m_n
     */
    cuda_array1i_t device_pchk_row_non_zeros;
    /*! Matrix where each row (representing a check m) contains the position (n)
     * of non-zero elements in the parity check matrix H (at that row of H).
     *
     * Extra space at the end of rows is padded with zeros/uninitalized.
     *
     * device_pchk_row_non_zero can be used to determine end of each row
     */
    cuda_matrixi_t device_pchk_row_non_zeros_pos;
    /*! Matrix where each row (representing a check m) contains the value (in
     * GF_q) of non-zero elements in the parity check matrix H (at that row of
     * H).
     *
     * Extra space at the end of rows is padded with zeros/uninitalized.
     *
     * device_pchk_row_non_zero can be used to determine end of each row
     */
    ::cuda::matrix<GF_q> device_pchk_row_non_zeros_val;

    /*! \brief Maximum number of non-zero elements in a column of the parity
     * check matrix.
     */
    int max_pchk_col_non_zeros;
    /*! \brief Array containing number of non-zero elements in each column of
     * parity matrix h_m_n
     */
    cuda_array1i_t device_pchk_col_non_zeros;
    /*! Matrix where each row (representing a codeword bit n) contains the
     * position (m) of non-zero elements in the parity check matrix H (at the
     * nth col of H).
     *
     * Extra space at the end of rows is padded with zeros/uninitalized.
     *
     * device_pchk_col_non_zeros can be used to determine end of each
     * row
     */
    cuda_matrixi_t device_pchk_col_non_zeros_pos;
    /*! Matrix where each row (representing a codeword bit n) contains the value
     * (in GF_q) of non-zero elements in the parity check matrix H (at that nth
     * col of H).
     *
     * Extra space at the end of rows is padded with zeros/uninitalized.
     *
     * device_pchk_col_non_zeros can be used to determine end of each row
     */
    ::cuda::matrix<GF_q> device_pchk_col_non_zeros_val;
};

} // namespace libcomm

#endif /* SUM_PROD_ALG_GDL_H_ */
