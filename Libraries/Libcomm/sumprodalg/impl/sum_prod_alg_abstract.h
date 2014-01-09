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

#ifndef SUM_PROD_ALG_ABSTRACT_H_
#define SUM_PROD_ALG_ABSTRACT_H_

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "sumprodalg/sum_prod_alg_inf.h"
#include <limits>

namespace libcomm {

/*! \brief Sum Product Algorithm(SPA) implementation
 *
 * Currently 2 types of the SPA: trad and gdl
 * The trad version computes the probabilities for the r__mxn's by computing
 * all the possible combinations of info symbols that satisfy the check node.
 * This can be very expensive computationally (especially when GF(q>2)
 * but it easy to code and understand.
 * The gdl version uses the fact that these probabilities can be grouped differently
 * using the distributive law and hence be computed much faster. The version
 * that is implemented here is based on Declercqs and Fossorier's 2006 paper:
 * Decoding Algorithms for Nonbinary LDPC Codes over GF(q)
 */
template <class GF_q, class real = double> class sum_prod_alg_abstract : public sum_prod_alg_inf<
      GF_q, real> {
public:

   /*! \name Type definitions */
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1i_t> array1vi_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

   /*! \brief constructor
    * initialise the main variables
    */
   sum_prod_alg_abstract(int n, int m, const array1vi_t& non_zero_col_pos,
         const array1vi_t& non_zero_row_pos,
         const libbase::matrix<GF_q>& pchk_matrix) :
      length_n(n), dim_m(m), M_n(non_zero_col_pos), N_m(non_zero_row_pos)
      {
      //default values for clipping method
      this->almostzero = real(1E-100);
      this->clipping_method = 0;

      this->marginal_probs.init(m, n);

      int non_zeros = 0;
      int pos = 0;

      for (int loop_m = 0; loop_m < this->dim_m; loop_m++)
         {
         non_zeros = this->N_m(loop_m).size();
         for (int loop_n = 0; loop_n < non_zeros; loop_n++)
            {
            pos = this->N_m(loop_m)(loop_n) - 1;//we count from zero;

            this->marginal_probs(loop_m, pos).val = pchk_matrix(loop_m, pos);
            }
         }
      }
   /*! \brief default destructor
    *
    */
   virtual ~sum_prod_alg_abstract()
      {
      //nothing to do
      }

   /*! \brief initialise the Sum Product algorithm with the relevant probabilities
    *
    */
   virtual void spa_init(const array1vd_t& ptable)=0;
   /*! \brief this returns the type of the Sum Product algorithm
    *
    */
   virtual std::string spa_type()=0;

   /*! \brief set the way the algorithm should deal with
    * clipping, ie replacing probabilities below a certain value
    */
   void set_clipping(std::string clipping_type, real almost_zero)
      {
      if ("zero" == clipping_type)
         {
         this->clipping_method = 0;
         }
      else
         {
         this->clipping_method = 1;
         }
      this->almostzero = almost_zero;

      }

   /*!\brief returns the type of clipping used
    *
    */
   std::string get_clipping_type()
      {
      std::string clipping_type;
      if (this->clipping_method == 1)
         {
         clipping_type = "clip";
         }
      else
         {
         clipping_type = "zero";
         }
      return clipping_type;
      }

   /*!\brief returns the value of almostzero used in the clipping method
    *
    */
   real get_almostzero()
      {
      return this->almostzero;
      }

   /*!\brief carry out one iteration of the SPA
    * This method will carry out the horizontal and vertical step
    * of the SPA and store the result in the ro vector
    */
   void spa_iteration(array1vd_t& ro);

   /*! brief Perform the desired clipping
    *
    */
   void perform_clipping(real& num)
      {
      if (1 == this->clipping_method)
         {
         //use standard clipping
         if (num < this->almostzero)
            {
            num = this->almostzero;
            }
         }
      else
         {
         //use zero clipping
         if (num <= real(0.0))
            {
            num = this->almostzero;
            }
         }
      }

protected:
   /*! \brief carries out the horizontal step of SPA
    * The r_mxn probabilities are computed
    */
   virtual void compute_r_mn(int m, int n, const array1i_t & tmpN_m)=0;
   /*! \brief carried out the horizontal step of the SPA
    * the q_mxn probabilities are computed
    */
   virtual void compute_q_mn(int m, int n, const array1i_t & M_n)=0;

private:
   void compute_probs(array1vd_t& ro);
   void print_marginal_probs(std::ostream& sout);
   void print_marginal_probs(int col, std::ostream& sout);

protected:

   /*! \name Data structures
    * LDPC specific datastructure used by the Sum-Product Algorithm
    */

   /* see MacKay's Information Theory, Inference and Learning Algs (2003, ch 47.3,pp 559-561)
    * for a proper definition of the following variables.
    */

   //!this struct holds the probabilities that check m is satisfied if symbol n of the received word is
   //fixed at symbols and the other symbols(<>n) have separable distributions given by q_mxn
   struct marginals {
      array1d_t q_mxn;
      array1d_t qmn_conv;//! this holds the fast FFT transforms of the q_mxns
      array1d_t r_mxn;
      GF_q val; //this holds the non_zero entry at position (m,n)
   };
   // @}

   //the number of cols
   int length_n;
   //the number of rows
   int dim_m;

   array1vd_t received_probs;

   //the positions of the non-zero entries per col
   array1vi_t M_n;

   //the positions of the non-zero entries per row
   array1vi_t N_m;

   //! this matrix holds the r_mxn probabilities
   libbase::matrix<marginals> marginal_probs;

   //! the clipping method used
   // 0-replace 0 with almostzero
   // 1-replace all values below almostzero with almostzero
   int clipping_method;

   //! this is the value we assign to zero probs
   real almostzero;

};

} // end namespace

#endif /* SUM_PROD_ALG_ABSTRACT_H_ */
