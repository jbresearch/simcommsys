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

#ifndef SUM_PROD_ALG_TRAD_H_
#define SUM_PROD_ALG_TRAD_H_

#include "sum_prod_alg_abstract.h"

namespace libcomm {
template <class GF_q, class real = double>
class sum_prod_alg_trad : public sum_prod_alg_abstract<GF_q, real> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1i_t> array1vi_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

   sum_prod_alg_trad(int n, int m, const array1vi_t& non_zero_col_pos,
         const array1vi_t& non_zero_row_pos,
         const libbase::matrix<GF_q>& pchk_matrix) :
      sum_prod_alg_abstract<GF_q, real>::sum_prod_alg_abstract(n, m,
            non_zero_col_pos, non_zero_row_pos, pchk_matrix)
      {
      }
   virtual ~sum_prod_alg_trad()
      {
      //nothing to do
      }
   void spa_init(const array1vd_t& ptable);
   void compute_r_mn(int m, int n, const array1i_t & tmpN_m);
   void compute_q_mn(int m, int n, const array1i_t & M_n);

   std::string spa_type()
      {
      return "trad";
      }
};

}

#endif /* SUM_PROD_ALG_TRAD_H_ */
