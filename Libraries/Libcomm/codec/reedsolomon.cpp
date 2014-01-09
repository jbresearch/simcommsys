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

/*
 * reedsolomon.cpp
 *
 *  Created on: 26-Jun-2009
 *      Author: swesemeyer
 */

#include "reedsolomon.h"
#include "linear_code_utils.h"
#include <cmath>
#include <sstream>
#include <iostream>

namespace libcomm {
using libbase::matrix;

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

//implementation of the relevant codec methods

template <class GF_q>
void reedsolomon<GF_q>::do_encode(const array1i_t & source, array1i_t & encoded)
   {
   libbase::linear_code_utils<GF_q, double>::encode_cw(this->gen_ref_matrix,
         source, encoded);
   }

template <class GF_q>
void reedsolomon<GF_q>::do_init_decoder(const array1vd_t & ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == this->num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == this->output_block_size());
   // Keep the likelihoods for future reference
   this->received_likelihoods = ptable;
   }

template <class GF_q>
void reedsolomon<GF_q>::do_init_decoder(const array1vd_t & ptable, const array1vd_t& app)
   {
   // Start by setting receiver statistics
   do_init_decoder(ptable);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(app.size() > 0);
   assertalways(app(0).size() == this->num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(app.size() == this->input_block_size());
   // Update likelihoods with prior statistics for info symbols
   for (int i = 0; i < this->dim_k; i++)
      {
      //the last k entries of the received word are the info symbols
      this->received_likelihoods(this->dim_pchk + i) *= app(i);
      }
   }

template <class GF_q>
void reedsolomon<GF_q>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   //determine the most likely symbol
   hd_functor(this->received_likelihoods, this->received_word_hd);
#if DEBUG>=2
   this->received_word_hd.serialize(std::cout, ',');
   std::cout << std::endl;
#endif

   //we use the PGZ algorithm for decoding General BCH codes (note that RS codes are
   //narrow-sense BCH codes.
   //see
   // http://en.wikipedia.org/wiki/BCH_code#Peterson_Gorenstein_Zierler_algorithm
   //for details.

   /*
    * The algorithm works as follows:
    * 1) calculate the syndrome of the received word
    * 2) calculate the error locator polynominal
    * 3) calculate the roots of the polynomial to get the error positions
    * 4) calculate the error-values at these locations.
    *
    */

   // in case of decoding failure we simply return the received probabilities
   ro = this->received_likelihoods;

   // Calculate the syndrome of the received word
   // (and flag whether we decoded successfully)
   libbase::vector<GF_q> syndrome_vec;
   const bool dec_success =
         libbase::linear_code_utils<GF_q, double>::compute_syndrome(
               this->pchk_matrix, this->received_word_hd, syndrome_vec);

#if DEBUG>=2
   std::cout << std::endl << "The received word is given by:" << std::endl;
   this->received_word_hd.serialize(std::cout, ',');
   std::cout << std::endl << "Its syndrome is given by:" << std::endl;
   syndrome_vec.serialize(std::cout, ',');
#endif
   if (dec_success)
      {
      // HD word must be correct, so set posteriors from this
      ro = double(0);
      for (int i = 0; i < this->length_n; i++)
         ro(i)(this->received_word_hd(i)) = double(1);
      }
   else //do some error correction as the syndrome is non-zero
      {
      //we can correct t errors and dmin=n-k+1.
      //note that we have dim_pck=(n-k)>=2t
      int t = (this->length_n - this->dim_k) / 2;

      //the syndrome matrix
      matrix<GF_q> syndrome_matrix;
      //its REF form
      matrix<GF_q> syndrome_ref_matrix;
      //the determinant of the syndrome matrix
      GF_q det = GF_q(1);

      //this is now the PGZ algorithm whose aim it is to
      //determine the error locator polynomial of the form
      // \lambda(x)=1 + \lambda_1 x +\lambda_2 x^2 + .. + \lambda_w x^w
      //where w<=t.
      do {
         //we now generate the t*(t+1) syndrome matrix
         /*
          *    [ s_1    s_2     s_3   ...     s_t   ]
          *    [ s_2    s_3     s_4   ...   s_{t+1} ]
          *  S=[ s_3    s_4     s_5   ...   s_{t+2} ]
          *    [ ...    ...     ...   ...     ...   ]
          *    [ s_t  s_{t+1} s_{t+2} ...  s_{2t-1} ]
          *
          */
         syndrome_matrix.init(t, t + 1);
         for (int rows = 0; rows < t; rows++)
            {
            for (int cols = 0; cols < t; cols++)
               {
               syndrome_matrix(rows, cols) = syndrome_vec(cols + rows);
               }
            }
         //this is C_{tx1}=(s_{t+1},s_{t+2}, ... ,s_{2t-1}]^t
         //we just stick this at the end of the syndrome matrix
         for (int rows = 0; rows < t; rows++)
            {
            syndrome_matrix(rows, t) = syndrome_vec(t + rows);
            }
         //compute the determinant of the syndrome matrix
         //which will in fact compute the solution to the following
         //system
         // S * [\lambda_1, \lambda_2,...,\lambda_t]^t=C_{tx1}^t
         syndrome_ref_matrix = syndrome_matrix.reduce_to_ref();

#if DEBUG>=2
         std::cout << std::endl << "The syndrome matrix is given by:" << std::endl;
         syndrome_matrix.serialize(std::cout, '\n');
         std::cout << std::endl << "The syndrome matrix in REF is given by:" << std::endl;
         syndrome_ref_matrix.serialize(std::cout, '\n');
#endif

         det = GF_q(1);
         for (int diag = 0; diag < t; diag++)
            {
            det *= syndrome_ref_matrix(diag, diag);
            }
         if (0 == det)
            {
            //empty error locator polynomial
            t--;
            }
         } while ((GF_q(0) == det) && (t >= 1));

      //only start decoding if det!=0
      if (GF_q(0) != det)
         {
         //We can now read off the coefficients of the error locator polynomial
         libbase::vector<GF_q> error_loc_poly;
         error_loc_poly.init(t + 1);
         error_loc_poly(0) = 1;
         for (int rows = 1; rows <= t; rows++)
            {
            error_loc_poly(rows) = GF_q(syndrome_ref_matrix(t - rows, t));
            }
         //we now want to factor the error locator polynomial as follows:
         //\lambda(x)=(X_w x +1)(X_{w-1} x +1) ... (X_1 x +1)
         //          =\lamba_w x^w+\lamba_{w-1} x^{w-1}+...+\lambda_1 x + 1
         //the X_i indicate the error positions of the received word as follows:
         //Suppose that errors happened at positions, j_1, j_2, ..,j_w then
         //X_i=\alpha^{j_i}.
#if DEBUG>=2
         std::cout
         << std::endl << "The coeffs of the error locator polynomial are given by:" << std::endl;
         error_loc_poly.serialize(std::cout, ',');
#endif

         //Use brute force and horner's scheme to determine the roots.
         //we can stop as soon as we have found t=deg(\lambda(x)) roots
         array1i_t error_pos;
         error_pos.init(t);
         GF_q alpha = GF_q(2); //represents \alpha
         int counter = 0;
         GF_q pow_alpha = GF_q(1); // represent \alpha^counter;
         int rootsfound = 0;
         while ((rootsfound < t) && (counter < this->length_n))
            {
            GF_q tmp_val = error_loc_poly(t);
            for (int j = t; j > 0; j--)
               {
               tmp_val = tmp_val * pow_alpha + error_loc_poly(j - 1);
               }
            if (tmp_val == GF_q(0))
               {
               //we have found a root, \beta=\alpha^s of the error locator polynomial, eg say
               //(X_1 \beta +1)=0 then X_1=\frac{1}{\beta}=\alpha^{-s}
               //Now X_1=\alpha^{j_1} this means that j_1=-s and as we only deal with powers between
               // 0 and q-1 we need to set j_1 to either 0 or (q-1)-s
               int pos_inv = this->length_n - counter;
               if (pos_inv == this->length_n)
                  {
                  pos_inv = 0;
                  }
               error_pos(rootsfound) = pos_inv;
               rootsfound++;
               }
            //increment the counter
            counter++;
            //up the power
            pow_alpha *= alpha;
            }

#if DEBUG>=2
         if (rootsfound > 0)
            {
            std::cout << std::endl << "We found roots at:" << std::endl;
            for (int loop1 = 0; loop1 < rootsfound; loop1++)
               {
               std::count << error_pos(loop1) << ", ";
               }
            }
#endif
         //only continue if we found some roots...
         if (rootsfound != 0)
            {
            //we have found some roots and hence the error locations
            //We can now work out the error values
            matrix<GF_q> error_mat;
            matrix<GF_q> error_ref_mat;
            error_mat.init(this->dim_pchk, rootsfound + 1);
            for (int rows = 0; rows < this->dim_pchk; rows++)
               {
               for (int cols = 0; cols < rootsfound; cols++)
                  {
                  error_mat(rows, cols) = this->pchk_matrix(rows,
                        error_pos(cols));
                  }
               }
            for (int rows = 0; rows < this->dim_pchk; rows++)
               {
               error_mat(rows, rootsfound) = syndrome_vec(rows);
               }
            //reduce to REF to obtain the error values
            error_ref_mat = error_mat.reduce_to_ref();

#if DEBUG>=2
            std::cout << std::endl << "The error matrix is given by:" << std::endl;
            error_mat.serialize(std::cout, '\n');
            std::cout << std::endl << "The error matrix in REF is given by:" << std::endl;
            error_ref_mat.serialize(std::cout, '\n');
#endif

            //we only have a consistent solution if the following value is 0
            if (error_ref_mat(rootsfound, rootsfound) == GF_q(0))
               {
               libbase::vector<GF_q> tmp_received_hd;
               tmp_received_hd = this->received_word_hd;

               //work out the proper code word
               for (int rows = 0; rows < rootsfound; rows++)
                  {
                  int col = error_pos(rows);
                  GF_q tmp_val = GF_q(tmp_received_hd(col))
                        - error_ref_mat(rows, rootsfound);
                  tmp_received_hd(col) = tmp_val;
                  }
#if DEBUG>=2
               std::cout << "This is the word we should have received:" << std::endl;
               tmp_received_hd.serialize(std::cout, ',');
               std::cout << std::endl;
#endif
               // decoded HD word is consistent, so set posteriors from this
               ro = double(0);
               for (int i = 0; i < this->length_n; i++)
                  ro(i)(tmp_received_hd(i)) = double(1);
               }
            }
         }
      }
   // Set input-referred posteriors from output-refered ones
   ri = ro.extract(this->dim_pchk, this->dim_k);

#if DEBUG>=2
   std::cout << std::endl << "The decoded word is given by:" << std::endl;
   decoded.serialize(std::cout, ',');
   std::cout << std::endl;
#endif
   }

template <class GF_q>
std::string reedsolomon<GF_q>::description() const
   {

   std::ostringstream sout;
   sout << "RS code [" << this->length_n << ", " << this->dim_k << "] ";

   libbase::trace << "Its parity check matrix is:" << std::endl;

   this->pchk_matrix.serialize(libbase::trace, ' ');
#if DEBUG>=1
   libbase::trace << "Its parity check matrix in REF is:" << std::endl;
   this->pchk_ref_matrix.serialize(libbase::trace, ' ');
#endif

   libbase::trace << "Its generator matrix in REF format is:" << std::endl;
   this->gen_ref_matrix.serialize(libbase::trace, ' ');
   return sout.str();
   }

template <class GF_q>
void reedsolomon<GF_q>::checkParams(int length, int dim)
   {

   //ensure the length makes sense
   //only consider RS codes that are narrow-sense BCH codes (ie n=q-1) for the moment
   //TODO extend this to singly-extended RS codes
   int maxlength = GF_q::elements() - 1;

   assertalways(length == maxlength);

   this->length_n = length;

   //ensure the dimension is sensible
   assertalways(dim < length);
   assertalways(dim >= 1);
   this->dim_k = dim;
   }

template <class GF_q>
void reedsolomon<GF_q>::init()
   {
   /*
    * Assume the multiplicative group GF(2^m)\{0}=<alpha>.
    *
    *
    * We consider the generator polynomial g(x)=(x-1)(x-alpha)(x-alpha^2)...(x-alpha^(2t-1)) where t>=1
    * Write g(x)=g_0+g_1*x+g_2*x^2+...+g_{2t-1}*x^{2t}
    * Set n-k=deg(g) where n=q-1, ie k=q-1-(deg(g))=q-1-2t=q-(2t+1)
    * Consider the following information polynomial of the form
    * u(x)=u_0+u_1x+...+u_(k-1)x^(k-1) with u_i in GF(2^m) then there are clearly
    * 2^(k*m) such polynomials.
    *
    * Using u(x) defined as above, we can compute v(x)= u(x)*g(x) where deg(v)<=deg(u)+deg(g)=k-1+(n-k)=n-1
    *
    * The coefficients of v(x) yield the code generated by g(x).
    *
    * Its generator matrix is clearly given by the following matrix:
    *
    *   [g_0   g_1 ... g_{n-k-2} g_{n-k-1}  g_{n-k}       0     0    0    ...  0    ]
    *   [ 0    g_0 ... g_{n-k-3} g_{n-k-2} g_{n-k-1}  g_{n-k}   0    0    ...  0    ]
    *   [ 0     0  ...                                                    ...  0    ]
    * G=[ .                                                                         ]
    *   [ .                                                                         ]
    *   [ .                                                                         ]
    *   [ 0     0  ...                                   g_0   g_1  g_2 ... g_{n-k} ]
    *
    * Using the Singleton-bound, it is clear that d<=n-k+1=(q-1)-(q-(2t+1))+1=2t-1
    *
    * Note that we can turn G into an extended Reed-Solomon code by simply adding another column that is the sum of all the
    * previous columns. let p=g_0+g_1+...+g_{n-k} then the generator matrix for the extended RS code is simply:
    *
    *   [g_0   g_1 ... g_{n-k-2} g_{n-k-1}  g_{n-k}       0     0    0    ...  0    p]
    *   [ 0    g_0 ... g_{n-k-3} g_{n-k-2} g_{n-k-1}  g_{n-k}   0    0    ...  0    p]
    *   [ 0     0  ...                                                    ...  0    p]
    * G=[ .                                                                         p]
    *   [ .                                                                         p]
    *   [ .                                                                         p]
    *   [ 0     0  ...                                   g_0   g_1  g_2 ... g_{n-k} p]
    *
    *
    * Looking at v(x)=u(x)g(x) we see that v(y)=0 when y={1,alpha,...,alpha^(2t-1)}, ie the roots of g(x).
    *
    * Consider the following matrix now:
    *
    *   [1    alpha           alpha^2             alpha^3      ...      alpha^(n-3)          alpha^(n-1)     ]
    *   [1   alpha^2        (alpha^2)^2         (alpha^3)^2    ...   (alpha^(n-3))^2       (alpha^(n-1)^2    ]
    * H=[.                                                     ...                                           ]
    *   [.                                                     ...                                           ]
    *   [.                                                     ...                                           ]
    *   [1  alpha^(2t-1)  (alpha^2)^(2t-1)  (alpha^3)^(2t-1)   ... (alpha^(n-3))^(2t-1)  (alpha^(n-1))^(2t-1)]
    *
    * Then it is clear that H*v=0
    *
    * The BCH bound now guarantees that the minimum distance of the code is d>=2t-1. So n-k>=2t-1.
    * Now the dimension of the code for which H is the parity check matrix is k=q-2t+1 and by the Singleton bound
    * d<=n-k+1=(q-1)-(q-2t+1)+1=2t-1 and hence d=2t-1.
    *
    * we can construct the parity check matrix of an extended RS code with [q,k,d+1] as follows (see Lin-p240):
    *
    *    [1    alpha           alpha^2             alpha^3      ...      alpha^(n-3)          alpha^(n-1)      1]
    *    [1   alpha^2        (alpha^2)^2         (alpha^3)^2    ...   (alpha^(n-3))^2       (alpha^(n-1)^2     0]
    * H'=[.                                                     ...                                            0]
    *    [.                                                     ...                                            0]
    *    [.                                                     ...                                            0]
    *    [1  alpha^(2t-1)  (alpha^2)^(2t-1)  (alpha^3)^(2t-1)   ... (alpha^(n-3))^(2t-1)  (alpha^(n-1))^(2t-1) 0]
    *
    * We will use the parity check matrix construction to obtain the required parity check matrix for the code
    * and then use standard matrix manipulation to obtain the generator matrix for the code.
    * in other words we use the following equalities if G=(P|I) then H=(I|-P^t) where P^t is the transpose of the matrix P.
    * since G*H^t=(P|I)*(I|-P^t)^t=P-(-P^t)^t)=P-P=0
    *
    */

   //what's the dimension of the parity check matrix
   this->dim_pchk = (this->length_n - this->dim_k);
   this->pchk_matrix.init(dim_pchk, this->length_n);

   int alpha = GF_q(2); //2 in binary is 10 which represents the generating element alpha
   int powerOfAlpha = GF_q(1); //this is alpha^0=1;

   //Determine whether we are dealing with an extended RS code
   bool extendedRS = (this->length_n == GF_q::elements());

   int codelength = (extendedRS ? this->length_n - 1 : this->length_n);

   //construct the parity check matrix H

   for (int n = 0; n < codelength; n++)
      {
      int tmp = powerOfAlpha;
      for (int k = 0; k < dim_pchk; k++)
         {
         //work out the appropriate power
         this->pchk_matrix(k, n) = GF_q(tmp);
         tmp = GF_q(tmp) * GF_q(powerOfAlpha);
         }
      powerOfAlpha = GF_q(powerOfAlpha) * GF_q(alpha);
      }

   if (extendedRS)
      {
      //We actually want H', so add the last column needed for H'
      this->pchk_matrix(0, codelength - 1) = 1;
      for (int k = 1; k < this->dim_pchk; k++)
         {
         this->pchk_matrix(k, codelength - 1) = 0;
         }
      }

   //turn it into RowEchelonForm to make the encoding
   //easier to debug

   this->pchk_ref_matrix.init(this->dim_pchk, this->length_n);
   this->pchk_ref_matrix = this->pchk_matrix.reduce_to_ref();

   //extract the -P^t part of H'=(I|-P^t)
   matrix<GF_q> tmp_p_transpose;
   tmp_p_transpose.init(this->dim_pchk, (this->length_n - this->dim_pchk));

   for (int loop = this->dim_pchk; loop < this->length_n; loop++)
      {
      tmp_p_transpose.insertcol(this->pchk_ref_matrix.extractcol(loop),
            loop - this->dim_pchk);
      }

   //this will hold P
   matrix<GF_q> tmp_p_transposed;
   tmp_p_transposed.init(this->dim_k, this->length_n - this->dim_k);
   //now transpose yourself
   tmp_p_transposed = tmp_p_transpose.transpose();
   matrix<GF_q> id_k = libbase::matrix<GF_q>::eye(this->dim_k);

   //Construct the generator matrix
   this->gen_ref_matrix.init(this->dim_k, this->length_n);

   //insert the transposed matrix
   for (int loop = 0; loop < (this->length_n - this->dim_k); loop++)
      {
      this->gen_ref_matrix.insertcol(tmp_p_transposed.extractcol(loop), loop);
      }

   //now add the identity matrix
   int counter = 0;
   for (int loop = (this->length_n - this->dim_k); loop < this->length_n;
         loop++)
      {
      this->gen_ref_matrix.insertcol(id_k.extractcol(counter), loop);
      counter++;
      }
   }

/*! serialization of the codec information
 * This method outputs the following format
 *
 * reedsolomon<gfq>
 * n
 * k
 *
 * where
 * q is the size of the finite field, ie GF(q)
 * n is the length of the code
 * k is its dimension
 *
 */
template <class GF_q>
std::ostream& reedsolomon<GF_q>::serialize(std::ostream& sout) const
   {
   // format version
   sout << "# Length of the code (n)" << std::endl;
   sout << this->length_n << std::endl;
   sout << "# Dimension of the code (k)" << std::endl;
   sout << this->dim_k << std::endl;
   return sout;
   }

// object serialization - loading
/*! loading of the serialized codec information
 * This method oxpects the following format
 *
 * reedsolomon
 * n
 * k
 * m
 *
 * where
 * n is the length of the code
 * k is its dimension
 * m determines the size of the finite field, eg GF(2^m)
 * note that we must have 1<k<n<2^m+1 and m<=10
 *
 */

template <class GF_q>
std::istream& reedsolomon<GF_q>::serialize(std::istream& sin)
   {
   assertalways(sin.good());

   int length, dim;
   //get the length
   sin >> libbase::eatcomments >> length >> libbase::verify;
   //get the dimension;
   sin >> libbase::eatcomments >> dim >> libbase::verify;
   //initialise the codec with this information
   this->checkParams(length, dim);
   init();
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

/* Serialization string: reedsolomon<type>
 * where:
 *      type = gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
      template class reedsolomon<type>; \
      template <> \
      const serializer reedsolomon<type>::shelper( \
            "codec", \
            "reedsolomon<" BOOST_PP_STRINGIZE(type) ">", \
            reedsolomon<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, GF_TYPE_SEQ)

} // end namespace
