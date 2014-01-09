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

#include "sum_prod_alg_abstract.h"

namespace libcomm {
// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class GF_q, class real> void sum_prod_alg_abstract<GF_q, real>::spa_iteration(
      array1vd_t& ro)
   {
   //carry out the horizontal step
   //this uses the description of the algorithm as given by
   //MacKay in Information Theory, Inference and Learning Algorithms(2003)
   //on page 560 - chapter 47.3

   // r_mxn(0)=\sum_{x_n'|n'\in N(m)\n'} ( P(z_m=0|x_n=0) * \prod_{n'\in N(m)\n}q_mxn(x_{n') )
   // Essentially, what we are doing is the following:
   // Assume x_n=0
   // we need to sum over all possibilities that such that the parity check is satisfied, ie =0
   // if the parity check is satisfied the conditional probability is 1 and 0 otherwise
   // so we are simply adding up the products for which the parity check is satisfied.

   //the number of symbols in N_m, eg the number of variables that participate in check m
   int size_N_m;

   //loop over all check nodes - the horizontal step
   for (int loop_m = 0; loop_m < this->dim_m; loop_m++)
      {
      // get the bits that participate in this check
      size_N_m = this->N_m(loop_m).size();
      for (int loop_n = 0; loop_n < size_N_m; loop_n++)
         {
         //this will compute the relevant r_nms fixing the x_n given by loop_n
         this->compute_r_mn(loop_m, loop_n, this->N_m(loop_m));
         }
      }

#if DEBUG>=2
   libbase::trace
   << std::endl << "After the horizontal step, the marginal matrix at col x is given by:" << std::endl;
   this->print_marginal_probs(3, libbase::trace);
#endif

   //this array holds the checks that use symbol n
   array1i_t M_n;
   //the number of checks in that array
   int size_M_n;

   //loop over all the bit nodes - the vertical step

   for (int loop_n = 0; loop_n < this->length_n; loop_n++)
      {
      M_n = this->M_n(loop_n);
      size_M_n = M_n.size().length();
      for (int loop_m = 0; loop_m < size_M_n; loop_m++)
         {
         this->compute_q_mn(loop_m, loop_n, M_n);
         }
      }
#if DEBUG>=2

   libbase::trace
   << "After the vertical step, the marginal matrix at col x is given by:" << std::endl;
   this->print_marginal_probs(3, libbase::trace);
#endif

   //compute the new probabilities for all symbols given the information in this iteration.
   //This will be used in a tentative decoding to see whether we have found a codeword
   this->compute_probs(ro);

#if DEBUG>=3
   libbase::trace
   << "The newly computed normalised probabilities are given by:" << std::endl;
   ro.serialize(libbase::trace, ' ');
#endif

   }

template <class GF_q, class real> void sum_prod_alg_abstract<GF_q, real>::compute_probs(
      array1vd_t& ro)
   {
   //ensure the output vector has the right length
   ro.init(this->length_n);

   //initialise some helper variables
   int num_of_elements = GF_q::elements();
   real a_n = real(0.0);
   int size_of_M_n = 0;
   int pos_m;
   for (int loop_n = 0; loop_n < this->length_n; loop_n++)
      {
      ro(loop_n) = this->received_probs(loop_n);
      size_of_M_n = this->M_n(loop_n).size();
      for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
         {
         for (int loop_m = 0; loop_m < size_of_M_n; loop_m++)
            {
            pos_m = this->M_n(loop_n)(loop_m) - 1;//we count from 0
            ro(loop_n)(loop_e) *= this->marginal_probs(pos_m, loop_n).r_mxn(
                  loop_e);
            }
         //Use appropriate clipping method
         perform_clipping(ro(loop_n)(loop_e));
         }
      //Note the following step is not strictly necessary apart from making the result
      //look neater - however it only adds a small overhead

      //normalise the result so that q_n_0+q_n_1=1
      a_n = ro(loop_n).sum();
      assertalways(a_n!=real(0.0));
      ro(loop_n) /= a_n;
      }
   }

template <class GF_q, class real> void sum_prod_alg_abstract<GF_q, real>::print_marginal_probs(
      std::ostream& sout)
   {
   int num_of_elements = GF_q::elements();
   bool used;
   for (int loop_m = 0; loop_m < this->dim_m; loop_m++)
      {
      sout << std::endl << "[";
      for (int loop_n = 0; loop_n < this->length_n; loop_n++)
         {
         sout << " <q=(";
         used = this->marginal_probs(loop_m, loop_n).q_mxn.size() > 0;
         if (used)
            {
            for (int loop_e = 0; loop_e < num_of_elements - 1; loop_e++)
               {
               sout << this->marginal_probs(loop_m, loop_n).q_mxn(loop_e)
                     << ", ";
               }
            sout << this->marginal_probs(loop_m, loop_n).q_mxn(num_of_elements
                  - 1);
            }
         else
            {
            sout << " n/a ";
            }
         sout << "), q_conv=(";
         if (used)
            {
            for (int loop_e = 0; loop_e < num_of_elements - 1; loop_e++)
               {
               sout << this->marginal_probs(loop_m, loop_n).qmn_conv(loop_e)
                     << ", ";
               }
            sout << this->marginal_probs(loop_m, loop_n).qmn_conv(
                  num_of_elements - 1);
            }
         else
            {
            sout << " n/a ";
            }
         sout << "), r=(";
         if (used)
            {
            for (int loop_e = 0; loop_e < num_of_elements - 1; loop_e++)
               {
               sout << this->marginal_probs(loop_m, loop_n).r_mxn(loop_e)
                     << ", ";
               }
            sout << this->marginal_probs(loop_m, loop_n).r_mxn(num_of_elements
                  - 1);
            }
         else
            {
            sout << "n/a ";
            }
         sout << "), val=(";
         if (used)
            {
            sout << this->marginal_probs(loop_m, loop_n).val;
            }
         else
            {
            sout << " n/a ";
            }
         sout << ")>";
         }
      sout << "]" << std::endl;
      }
   }
template <class GF_q, class real> void sum_prod_alg_abstract<GF_q, real>::print_marginal_probs(
      int col, std::ostream& sout)
   {
   int num_of_elements = GF_q::elements();
   int tmp_row;
   int tmp_col;
   sout << "only printing the necessary values for col=" << col;
   col--;//we count from 0
   array1i_t tmp_N_m;
   int num_of_elements_in_col = this->M_n(col).size();
   int num_of_elements_in_row = 0;

   for (int loop_m = 0; loop_m < num_of_elements_in_col; loop_m++)
      {
      tmp_row = this->M_n(col)(loop_m) - 1;
      sout << std::endl << "row=" << tmp_row + 1;
      sout << std::endl << "[";
      tmp_N_m = this->N_m(tmp_row);
      num_of_elements_in_row = tmp_N_m.size();
      for (int loop_n = 0; loop_n < num_of_elements_in_row; loop_n++)
         {
         tmp_col = tmp_N_m(loop_n) - 1;
         sout << std::endl << " <q=(";
         for (int loop_e = 0; loop_e < num_of_elements - 1; loop_e++)
            {
            sout << this->marginal_probs(tmp_row, tmp_col).q_mxn(loop_e)
                  << ", ";
            }
         sout << this->marginal_probs(tmp_row, tmp_col).q_mxn(num_of_elements
               - 1);
         bool used = this->marginal_probs(tmp_row, tmp_col).qmn_conv.size() > 0;
         if (used)
            {
            sout << "),\n q_conv=(";
            for (int loop_e = 0; loop_e < num_of_elements - 1; loop_e++)
               {
               sout << this->marginal_probs(tmp_row, tmp_col).qmn_conv(loop_e)
                     << ", ";
               }
            sout << this->marginal_probs(tmp_row, tmp_col).qmn_conv(
                  num_of_elements - 1);
            }
         sout << "),\n r=(";

         for (int loop_e = 0; loop_e < num_of_elements - 1; loop_e++)
            {
            sout << this->marginal_probs(tmp_row, tmp_col).r_mxn(loop_e)
                  << ", ";
            }
         sout << this->marginal_probs(tmp_row, tmp_col).r_mxn(num_of_elements
               - 1);

         sout << "), val=(";

         sout << this->marginal_probs(tmp_row, tmp_col).val;

         sout << ")>";
         }
      sout << "]" << std::endl;
      }
   }

} // end namespace

#include "gf.h"
#include "mpreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

using libbase::mpreal;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define REAL_TYPE_SEQ \
   (double)(logrealfast)(mpreal)

/* Serialization string: ldpc<type,real>
 * where:
 *      type = gf2 | gf4 ...
 *      real = double | logrealfast | mpreal
 */
#define INSTANTIATE(r, args) \
      template class sum_prod_alg_abstract<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
