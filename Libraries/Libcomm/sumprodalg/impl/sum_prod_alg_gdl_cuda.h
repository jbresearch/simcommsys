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
#include "matrix.h"
#include "vector.h"
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

    perms(ix, iy) = GF_q(ix) * GF_q(i_y);
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
    typedef libbase::cuda::matrix<int> cuda_matrixi_t;
    typedef libbase::cuda::matrix<real> cuda_matrixd_t;
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
                device_perms, num_of_elements);

        // this will copy over the parity check matrix from host to the device.
        device_parity_chk_matrix = pchk_matrix;

        device_qmn_row_indices.init(m, n);
        device_qmn_row_indices.fill(-1);

        // we first build qmn_row_indices on the host, then copy to device.
        // Easier since this operation is inherently serial (we have a counter
        // to keep track of current index) and also we need the tanner_edges var
        // computed during this process on host to allocate memory for qmn and
        // rmn matrices.
        libbase::matrix qmn_row_indices(m, n);

        // counts the number of edges in the Tanner graph of the code.
        // Tells us what the size of device_rmxn and device_qmn_conv should
        // be.
        int tanner_edges = 0;

        // Populate qmn_row_indices
        int non_zeros = 0;
        int pos = 0;
        for (int loop_m = 0; loop_m < this->dim_m; loop_m++) {
            non_zeros = non_zero_row_pos(loop_m).size();

            for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
                pos =
                    non_zero_row_pos(loop_m)(loop_n) - 1; // we count from zero;

                qmn_row_indices(loop_m, pos) = tanner_edges;
                tanner_edges++;
            }
        }

        // Copy qmn_row_indices to device
        device_qmn_row_indices = qmn_row_indices;

        // Allocate required memory for r_mxn and qmn_conv on device.
        device_r_mxn.init(tanner_edges, num_of_elements);
        device_qmn_conv.init(tanner_edges, num_of_elements);
    }
    virtual ~sum_prod_alg_gdl_cuda()
    {
        // nothing to do
    }

    // Overriden methods from sum_prod_alg_inf.
    void spa_init(const array1vd_t& ptable) override;
    void spa_iteration(array1vd_t& ro) override;
    std::string spa_type() override { return "gdl_cuda"; }
    void set_clipping(std::string clipping_type, real almost_zero) override;
    void std::string get_clipping_type() override;
    real get_almostzero() override;
    void perform_clipping(real& num) override;

    // Methods to organize computation of SPA
    void compute_r_mn(int m, int n, const array1i_t& tmpN_m);
    void compute_q_mn(int m, int n, const array1i_t& M_n);
    void compute_probs(array1vd_t& ro);

private:
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
    void compute_convs(array1d_t& conv_out, int pos1, int pos2);

private:
    /*! \brief this holds a look-up table of the finite field multiplication
     */
    cuda_matrixi_t device_perms;

    cuda_matrixd_t device_received_probs;

    //! \brief the parity check matrix itself
    libbase::cuda::matrix<GF_q> device_parity_chk_matrix;

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
    cuda_matrixd_t device_qmn_conv;
};

} // namespace libcomm

#endif /* SUM_PROD_ALG_GDL_H_ */
