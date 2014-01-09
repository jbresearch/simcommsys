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

#include "sum_prod_alg_gdl.h"
#include "gf.h"
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class GF_q, class real>
void sum_prod_alg_gdl<GF_q, real>::spa_init(const array1vd_t& recvd_probs)
   {

   int num_of_elements = GF_q::elements();
   real tmp_prob = real(0.0);
   real alpha = real(0.0);

   //initialise the marginal prob values
   //ensure we don't have zero probabilities
   //and normalise the probs at the same time
   this->received_probs.init(recvd_probs.size());
   for (int loop_n = 0; loop_n < this->length_n; loop_n++)
      {
      this->received_probs(loop_n).init(num_of_elements);
      alpha = real(0.0);
      for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
         {
         //Clipping HACK
         tmp_prob = recvd_probs(loop_n)(loop_e);
         this->perform_clipping(tmp_prob);
         this->received_probs(loop_n)(loop_e) = tmp_prob;
         alpha += tmp_prob;
         }
      assertalways(alpha!=real(0.0));
      this->received_probs(loop_n) /= alpha;
      }

#if DEBUG>=2
   libbase::trace << std::endl << "The first 5 normalised likelihoods are given by:" << std::endl;
   libbase::trace << this->received_probs.extract(0, 5);
#endif

   //this uses the description of the algorithm as given by

   //MacKay in Information Theory, Inference and Learning Algorithms(2003)
   //on page 560 - chapter 47.3

   //some helper variables
   int pos = 0;
   int non_zeros = 0;
   int h_m_n = 0;

   array1vd_t qmn_conv;

   //simply set q_mxn(0)=P_n(0)=P(x_n=0) and q_mxn(1)=P_n(1)=P(x_n=1)
   for (int loop_m = 0; loop_m < this->dim_m; loop_m++)
      {
      non_zeros = this->N_m(loop_m).size();
      for (int loop_n = 0; loop_n < non_zeros; loop_n++)
         {
         pos = this->N_m(loop_m)(loop_n) - 1;//we count from zero;
         h_m_n = this-> marginal_probs(loop_m, pos).val;
         this->marginal_probs(loop_m, pos).q_mxn = this->received_probs(pos);

         // If h_m_n is not 1 we need to permute the probs.
         if (1 != h_m_n)
            {
            //In fact the probability we are given are not for the x_i but for
            //the value h_m_n*xi hence all we need to do is copy the values into
            //the array with a slightly amended index:
            //probs(h_m_n*x)=received_prob(x) for all x in GF_q and 0!=h_m_n in GF_q.
            // Declerq&Fossorier: Decoding Algs for non-binary LDPC Codes over GF(q)
            for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
               {
               //perms(h_m_n)(loop)=GF_q(h_m_n)*GF_q(loop) - a look-up is quicker than a
               //computation (I hope)
               this ->marginal_probs(loop_m, pos).qmn_conv(this->perms(h_m_n)(
                     loop_e)) = this->received_probs(pos)(loop_e);
               }
            }
         else
            {
            //no permutation needed as h_m_n=1
            this->marginal_probs(loop_m, pos).qmn_conv = this->received_probs(
                  pos);
            }
         this ->compute_convs(this ->marginal_probs(loop_m, pos).qmn_conv, 0,
               num_of_elements - 1);
         }
      this->marginal_probs(loop_m, pos).r_mxn = 0.0;
      }

#if DEBUG>=2
   libbase::trace << " Memory Usage:\n ";
   libbase::trace << this->marginal_probs.size()
   * sizeof(sum_prod_alg_abstract<GF_q,real>::marginals) / double(1 << 20)
   << " MB" << std::endl;

   libbase::trace << std::endl << "The marginal matrix is given by:" << std::endl;
   this->print_marginal_probs(libbase::trace);
#endif

   }

template <class GF_q, class real>
void sum_prod_alg_gdl<GF_q, real>::compute_convs(array1d_t & conv_out,
      int pos1, int pos2)
   {
   //this is in fact the Hadamard transform using the butterfly property of
   //the fast Fourier transform.
   if ((pos2 - pos1) == 1)
      {
      real tmp1 = conv_out(pos1);
      real tmp2 = conv_out(pos2);
      conv_out(pos1) = tmp1 + tmp2;
      conv_out(pos2) = tmp1 - tmp2;
      }
   else
      {
      int midpoint = pos1 + (pos2 - pos1 + 1) / 2;
      this->compute_convs(conv_out, pos1, midpoint - 1);
      this->compute_convs(conv_out, midpoint, pos2);
      pos2 = midpoint;
      for (int loop1 = pos1; loop1 < midpoint; loop1++)
         {
         real tmp1 = conv_out(loop1);
         real tmp2 = conv_out(pos2);
         conv_out(loop1) = tmp1 + tmp2;
         conv_out(pos2) = tmp1 - tmp2;
         pos2++;
         }
      }
   }

//specialisation for GF(2)
template <>
void sum_prod_alg_gdl<libbase::gf2 , double>::compute_r_mn(int m, int n,
      const array1i_t & tmpN_m)
   {
   //the number of participating symbols
   int num_of_var_syms = tmpN_m.size();

   int pos_n = tmpN_m(n) - 1;//we count from 1;

   int pos_n_dash;

   double q_nm_conv_prod = 1.0;
   for (int loop2 = 0; loop2 < num_of_var_syms; loop2++)
      {
      if (loop2 != n)
         {
         pos_n_dash = tmpN_m(loop2) - 1;//we count from zero
         q_nm_conv_prod *= this->marginal_probs(m, pos_n_dash).qmn_conv(1);
         }
      }
   this->marginal_probs(m, pos_n).r_mxn(0) = 0.5 * (1.0 + q_nm_conv_prod);
   this->marginal_probs(m, pos_n).r_mxn(1) = 0.5 * (1.0 - q_nm_conv_prod);

   }

template <class GF_q, class real>
void sum_prod_alg_gdl<GF_q, real>::compute_r_mn(int m, int n,
      const array1i_t & tmpN_m)
   {
   //the number of participating symbols
   int num_of_var_syms = tmpN_m.size();
   int num_of_elements = GF_q::elements();

   int pos_n = tmpN_m(n) - 1;//we count from 1;
   //note the following should never be a division by zero!
   int h_m_n = this->marginal_probs(m, pos_n).val;

   int pos_n_dash;

   array1d_t q_nm_conv_prod;
   q_nm_conv_prod.init(num_of_elements);
   q_nm_conv_prod = 1.0;
   for (int loop2 = 1; loop2 < num_of_elements; loop2++)
      {
      for (int loop1 = 0; loop1 < num_of_var_syms; loop1++)
         {
         if (loop1 != n)
            {
            pos_n_dash = tmpN_m(loop1) - 1;//we count from zero

            //this uses the FFT of the q_mxn to work out the r_mn
            q_nm_conv_prod(loop2)
                  *= this->marginal_probs(m, pos_n_dash).qmn_conv(loop2);
            }
         }
      }

   //apply the FFT again to get the proper values
   this->compute_convs(q_nm_conv_prod, 0, num_of_elements - 1);

   /*
    * ensure that the values in q_nm_conv_prod make sense, ie
    * they should all be >0 and sum up to 1
    * Note we don't allow zero values either as no prob should be
    * completely 0.
    */
   real sum_qnm = real(0.0);
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      //Clipping HACK
      this->perform_clipping(q_nm_conv_prod(loop1));
      sum_qnm += q_nm_conv_prod(loop1);
      }

   //normalise them instead of simply dividing by the number of field elements.
   assertalways(sum_qnm != real(0.0));
   q_nm_conv_prod /= sum_qnm;

   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      //perms(h_m_n)(loop)=GF_q(h_m_n)*GF_q(loop) - a look-up is quicker than a
      //computation (I hope)
      this ->marginal_probs(m, pos_n).r_mxn(loop1) = q_nm_conv_prod(
            this->perms(h_m_n)(loop1));
      }
   }

template <class GF_q, class real>
void sum_prod_alg_gdl<GF_q, real>::compute_q_mn(int m, int n,
      const array1i_t & M_n)
   {
   //initialise some helper variables
   int num_of_elements = GF_q::elements();
   array1d_t q_mn(this -> received_probs(n));
   real a_nxm = q_mn.sum();//sum up the values in q_mn
   assertalways(a_nxm!=real(0));
   int m_dash = 0;
   int pos_m = M_n(m) - 1;//we count from 1;

   //compute q_mn(sym) = a_mxn * P_n(sym) * \prod_{m'\in M(n)\m} r_m'xn(0) for all sym in GF_q
   int size_of_M_n = M_n.size().length();

   for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
      {
      for (int loop_m = 0; loop_m < size_of_M_n; loop_m++)
         {
         if (m != loop_m)
            {
            m_dash = M_n(loop_m) - 1; //we start counting from zero

            q_mn(loop_e) *= this->marginal_probs(m_dash, n).r_mxn(loop_e);
            }
         }
      //Clipping HACK
      this->perform_clipping(q_mn(loop_e));
      }

   //normalise the q_mxn's so that q_mxn_0+q_mxn_1=1
   a_nxm = q_mn.sum();//sum up the values in q_mn

   if (a_nxm == real(0))
      {
      //show me the error
      q_mn = this -> received_probs(n);
      std::cerr << "received probs:" << q_mn;
      for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
         {
         for (int loop_m = 0; loop_m < size_of_M_n; loop_m++)
            {
            if (m != loop_m)
               {
               m_dash = M_n(loop_m) - 1; //we start counting from zero
               std::cerr << "q_mn(" << loop_e << ")=" << q_mn(loop_e) << " x "
                     << this->marginal_probs(m_dash, n).r_mxn(loop_e) << std::endl;
               q_mn(loop_e) *= this->marginal_probs(m_dash, n).r_mxn(loop_e);
               }
            }
         //Clipping HACK - just for error display purposes
         if (1 == this->clipping_method)
            {
            if (q_mn(loop_e) < this->almostzero)
               {
               std::cerr << "q_mn(" << loop_e
                     << ")<almostzero - setting it to almostzero="
                     << this->almostzero << std::endl;
               q_mn(loop_e) = this->almostzero;
               }
            }
         else
            {
            if (q_mn(loop_e) <= real(0.0))
               {
               std::cerr << "q_mn(" << loop_e
                     << ") is equal to zero - setting it to almostzero="
                     << this->almostzero << std::endl;
               q_mn(loop_e) = this->almostzero;
               }

            }
         }
      }
   assertalways(a_nxm!=real(0));
   q_mn /= a_nxm; //normalise
   //store the values
   this->marginal_probs(pos_m, n).q_mxn = q_mn;
   //compute the FFT and store it for the next iteration
   int h_m_n = this->marginal_probs(pos_m, n).val;
   for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
      {
      //perms(h_m_n)(loop)=GF_q(h_m_n)*GF_q(loop) - a look-up is quicker than a
      //computation (I hope)
      this ->marginal_probs(pos_m, n).qmn_conv(this->perms(h_m_n)(loop_e))
            = q_mn(loop_e);
      }
   this ->compute_convs(this ->marginal_probs(pos_m, n).qmn_conv, 0,
         num_of_elements - 1);

   }

} // end namespace

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
      template class sum_prod_alg_gdl<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
