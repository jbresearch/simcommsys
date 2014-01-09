/*!
 * \file
 *
 * Copyright (c) 2010 Stephan Wesemeyer
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

#include "sum_prod_alg_abstract.h"
#include <string>

namespace libcomm {

template <class GF_q, class real = double>
class sum_prod_alg_gdl : public sum_prod_alg_abstract<GF_q, real> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1i_t> array1vi_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

   /*! \brief constructor
    * This constructor calls the parent class but then
    * also creates a multiplication look-up table for the relevant
    * finite field. This is needed as multiplication in GF_q is fairly
    * expensive at the moment. Ideally, this look-up table should be moved
    * to the finite field implementation
    *
    */
   sum_prod_alg_gdl(int n, int m, const array1vi_t& non_zero_col_pos,
         const array1vi_t& non_zero_row_pos,
         const libbase::matrix<GF_q>& pchk_matrix) :
      sum_prod_alg_abstract<GF_q, real>::sum_prod_alg_abstract(n, m,
            non_zero_col_pos, non_zero_row_pos, pchk_matrix)
      {
      int num_of_elements = GF_q::elements();
      int non_zeros = 0;
      int pos = 0;
      for (int loop_m = 0; loop_m < this->dim_m; loop_m++)
         {
         non_zeros = this->N_m(loop_m).size();
         for (int loop_n = 0; loop_n < non_zeros; loop_n++)
            {
            pos = this->N_m(loop_m)(loop_n) - 1;//we count from zero;
            this->marginal_probs(loop_m, pos).qmn_conv.init(num_of_elements);
            this->marginal_probs(loop_m, pos).r_mxn.init(num_of_elements);
            }
         }

      this->perms.init(num_of_elements);
      this->perms(0).init(num_of_elements);
      this->perms(0) = 0; //note this is by convention and not used anywhere

      for (int loop1 = 1; loop1 < num_of_elements; loop1++)
         {
         this ->perms(loop1).init(num_of_elements);
         pos = 0;
         for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
            {
            this ->perms(loop1)(pos) = GF_q(loop_e) * GF_q(loop1);
            pos++;
            }
         }
      }
   virtual ~sum_prod_alg_gdl()
      {
      //nothing to do
      }
   void spa_init(const array1vd_t& ptable);
   void compute_r_mn(int m, int n, const array1i_t & tmpN_m);
   void compute_q_mn(int m, int n, const array1i_t & M_n);
   std::string spa_type()
      {
      return "gdl";
      }

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
   void compute_convs(array1d_t & conv_out, int pos1, int pos2);

private:
   /*! \brief this holds a look-up table of the finite field multiplication
    *
    */
   array1vi_t perms;

};

}

#endif /* SUM_PROD_ALG_GDL_H_ */
