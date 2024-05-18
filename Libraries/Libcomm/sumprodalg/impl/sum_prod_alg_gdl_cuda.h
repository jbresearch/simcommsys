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

#ifndef SUM_PROD_ALG_GDL_CUDA_H_
#define SUM_PROD_ALG_GDL_CUDA_H_

#include "../sum_prod_alg_inf.h"
#include "cuda/matrix.h"
#include "cuda/vector.h"
#include "matrix.h"
#include "vector.h"
#include <limits>
#include <string>

namespace libcomm
{

template <class GF_q, class real = double>
class sum_prod_alg_gdl_cuda : public sum_prod_alg_inf<GF_q, real>
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<real> array1d_t;
    typedef libbase::vector<int> array1i_t;
    typedef libbase::vector<array1i_t> array1vi_t;
    typedef libbase::vector<array1d_t> array1vd_t;
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
                          const libbase::matrix<GF_q>& pchk_matrix);

    virtual ~sum_prod_alg_gdl_cuda()
    {
        // nothing to do
    }

    // Overriden methods from sum_prod_alg_inf.
    void spa_init(const array1vd_t& ptable) override;
    void spa_iteration(array1vd_t& ro) override;
    std::string spa_type() override { return "gdl_cuda"; }

private:
    /*! \brief this is an n x |GF_q| size matrix that holds prior probability
     * distributions of each symbol in a codeword.
     */
    cuda_matrixd_t device_received_probs;
    /*! \brief this is an n x |GF_q| size matrix that holds posterior
     * probability distributions of each symbol in a codeword.
     */
    cuda_matrixd_t device_out_probs;

    /*! Matrix of indices that tell us row of device_qmn_conv that
     * contains prob distribution qmn for a particular (m, loop_n).
     * Also works for device_r_mxn since prob. distr. qmn and r_mxn have the
     * same size (size of GF(q) as there is one prob. for each element of GF(q))
     */
    cuda_matrixi_t device_mxn_row_idx_lut;
    cuda_matrixi_t device_nxm_row_idx_lut;

    /*! Each row of this matrix is a probability distribution r_mxn.
     * There is a row for each combination (m, n) such that bit n participates
     * in check m. The mapping between (m, n) and the rows is given by
     * device_mxn_row_idx_lut, device_nxm_row_idx_lut
     */
    cuda_matrixd_t device_r_mxn;
    /*! Each row of this matrix is a probability distribution q_mn (or more
     * accurately the Hadamard transform of the q_mn in usual SPA).
     *
     * There is a row for each combination (m, n) such that bit n participates
     * in check m. The mapping between (m, n) and the rows is given by
     * device_mxn_row_idx_lut, device_nxm_row_idx_lut
     */
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
};

} // namespace libcomm

#endif /* SUM_PROD_ALG_GDL_CUDA_H_ */
