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
#include "cuda/device_ptr.h"
#include "cuda/matrix.h"
#include "cuda/stream.h"
#include "cuda/vector.h"
#include "matrix.h"
#include "random.h"
#include "vector.h"
#include <array>
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
    typedef libbase::matrix<int> matrixi_t;
    typedef libbase::matrix<real> array2d_t;

    typedef ::cuda::vector<int> cuda_array1i_t;
    typedef ::cuda::matrix<int, false> cuda_matrixi_t;
    typedef ::cuda::matrix<real, false> cuda_matrixd_t;

    typedef sum_prod_alg_inf<GF_q, real> Base;
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
    void spa_init(const array2d_t& ptable) override;
    std::string spa_type() override { return "gdl"; }
    int get_iters() override { return this->num_iters; }

    void spa_iteration(libbase::vector<GF_q>& received_word) override;
    void decode(libbase::vector<GF_q>& received_word, int max_iters) override;

    void seedfrom(libbase::random& r) override;

private:
    /*! \name State variables */
    /*! \brief this is an n x |GF_q| size matrix that holds prior probability
     * distributions of each symbol in a codeword.
     */
    cuda_matrixd_t device_received_probs;
    /*! \brief this is an n x |GF_q| size matrix that holds posterior
     * probability distributions of each symbol in a codeword.
     */
    cuda_matrixd_t device_out_probs;

    /*! Matrix of indices that tell us row of device_qmn_conv that
     * contains prob distribution qmn for a particular (m, n).
     * Also works for device_r_mxn since prob. distr. qmn and r_mxn have the
     * same size (size of GF(q) as there is one prob. for each element of GF(q))
     *
     * If bit n does not participate in check m, i.e. h_mn = 0, then the
     * corresponding element of this matrix is -1 by convention.
     */
    cuda_matrixi_t device_qmn_row_mxn_indices;
    cuda_matrixi_t device_qmn_row_nxm_indices;

    /*! Each row of this matrix is a probability distribution r_mxn.
     * There is a row for each combination (m, n) such that bit n participates
     * in check m. The mapping between (m, n) and the rows is given by
     * device_qmn_row_indices
     */
    cuda_matrixd_t device_r_mxn;
    /*! Each row of this matrix is a probability distribution q_mn (or more
     * accurately the Hadamard transform of the q_mn in usual SPA).
     *
     * There is a row for each combination (m, n) such that bit n participates
     * in check m. The mapping between (m, n) and the rows is given by
     * device_qmn_row_indices
     */
    cuda_matrixd_t device_qmn_conv;

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
    /*! Matrix where each row (representing a pchk m) contains the position of
     * non-zero elements in the parity check matrix H (at that row of H).
     *
     * Extra space at the end of rows is padded with zeros/uninitalized.
     *
     * device_pchk_row_non_zero can be used to determine end of each row
     */
    cuda_matrixi_t device_pchk_row_non_zeros_pos;
    /*! Matrix where each row (representing a pchk m) contains the value (in
     * GF_q) of non-zero elements in the parity check matrix H (at that row
     * of H).
     *
     * Extra space at the end of rows is padded with zeros/uninitalized.
     *
     * device_pchk_row_non_zero can be used to determine end of each row
     */
    ::cuda::matrix<GF_q, false> device_pchk_row_non_zeros_val;

    /*! \brief Maximum number of non-zero elements in a column of the parity
     * check matrix.
     */
    int max_pchk_col_non_zeros;
    /*! \brief Array containing number of non-zero elements in each column of
     * parity matrix h_m_n
     */
    cuda_array1i_t device_pchk_col_non_zeros;
    /*! Vector containing the non-zero values of H_mn, in the same order as
     * the rows of device_q_mn_conv.
     *
     * In more detail, if the kth row of device_q_mn_conv corresponds to edge
     * v_j <-> c_i device_pchk_non_zeros_val[k] = H_ij
     */
    ::cuda::vector<GF_q> device_pchk_non_zeros_val;

    /*! \brief Stores the received codeword according to decoder. */
    ::cuda::vector<GF_q> device_received_word;
    /*! \brief Used when computing syndrome to check if an iteration has yielded
     * valid codeword or not. */
    ::cuda::vector<GF_q> device_syndrome;

    /*! \brief Set to true when iteration yields a valid codeword, false
     * otherwise. */
    ::cuda::device_ptr<bool> device_decode_success;
    /*! \brief Hard-decision box used when determining codeword from
     * probabilities. */
    ::cuda::device_ptr<
        basic_hard_decision<real, GF_q, ::cuda::vector_reference<real>>>
        hd_functor;

    //! Number of iterations used to decode last codeword
    int num_iters = 0;

    /*! \name Variables used only when decoding is done through calls to
     * spa_iteration().
     */
    /*! \brief Stores received codeword when one is found using spa_iteration().
     */
    libbase::vector<GF_q> received_word;
    /*! \brief Indicates whether a previous call to spa_iteration() has already
     * successfully found a codeword. */
    bool decode_success;

    /*! \brief Number of tanner edges in the pchk matrix
     */
    size_t tanner_edges;

    /*! \brief Device ID for the CUDA-enabled device used.
     */
    int device;

    /*! \brief Warp size for the CUDA-enabled device used.
     */
    int warpSize;

private:
    /*! \name Internal methods for a single SPA iteration */
    /*! \brief carry out one iteration of the SPA but do not copy results from
     * the GPU.
     * \return Whether or not a codeword has been found. */
    bool spa_iteration();
    void compute_r_mn();
    void compute_q_mn();
    void compute_probs();
};

} // namespace libcomm

#endif /* SUM_PROD_ALG_GDL_CUDA_H_ */
