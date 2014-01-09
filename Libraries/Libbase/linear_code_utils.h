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

#ifndef LINEARCODEUTILS_H_
#define LINEARCODEUTILS_H_

#include "matrix.h"

namespace libbase {
/*!
 *  \brief   Linear Block Code Helper Class
 *  \author  S Wesemeyer
 * This class is a helper class and provides utility methods for
 * linear block codes over GF(2^p) with p>=1
 */

template <class GF_q, class real = double>
class linear_code_utils {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<GF_q> array1gfq_t;
   typedef libbase::vector<array1d_t> array1dv_t;

   /*!This computes the dual code of the given matrix as follows:
    * 1) reduce orgMat to REF, call this matrix G
    * 2) Let G'=\pi(G) so that it is in the following format G'=(I|P)
    * 3) Let H'=(-P^t|I)
    * 4) Let H=\pi^{-1}(H')
    * H is then the generator matrix of the dual code of G. H is also
    * the parity check matrix of G.
    */
   static void compute_dual_code(const libbase::matrix<GF_q> & orgMat,
         libbase::matrix<GF_q> & dualCodeMatrix, array1i_t & systematic_perm);

   /*
    * !This computes the row space of a parity check matrix,
    * ie given an mxn matrix H, this method determines the linear
    * independent rows of H and returns a new matrix consisting only
    * of those rows.
    * Note this is mainly useful for LDPC codes that are defined by their
    * parity check matrix.
    */
   static void compute_row_dim(const libbase::matrix<GF_q> & parMat_H,
         libbase::matrix<GF_q> & maxRowSpace_H);

   /*
    * !This removes any zero columns from a matrix and returns a new matrix
    * without these columns.
    */
   static void remove_zero_cols(const libbase::matrix<GF_q> & mat_G,
         libbase::matrix<GF_q> noZeroCols_G);

   /*
    * !This encodes a codeword given a generator matrix
    */
   static void encode_cw(const libbase::matrix<GF_q> & mat_G, const array1i_t & source,
         array1i_t & encoded);

   /*
    * !This computes the syndrome of a received word - it returns true if the syndrome is the zero vector.
    *
    */
   static bool compute_syndrome(const libbase::matrix<GF_q> & parMat,
         const array1gfq_t & received_word_hd, array1gfq_t & syndrome);

   /*! This checks whether the generator matrix is in systematic form (eg G=(I|P))
    *
    */
   static bool is_systematic(const libbase::matrix<GF_q> & genMat);

   /*!
    * This creates a Hadamard matrix of size=2^m
    */
   static void create_hadamard(libbase::matrix<int>& hadMat, int m);

   /*!
    * This computes the Kroenecker product of 2 matrices
    * if A is an mxn matrix and B is a pxq matrix then their Kroenecker product is
    * an (m*p)(n*q) matrix. The result of the product is that each a_i_j in A is effectively
    * replaced by the matrix a_i_j*B.
    */
   static void compute_kronecker(const libbase::matrix<int>& A, const libbase::matrix<int>& B,
         libbase::matrix<int>& prod);

   /*! \brief convert a parity check matrix over GF(q) into the equivalent binary one
    *
    */
   /*
    static void convert_to_binary(const libbase::matrix<GF_q>& mat_in,
    libbase::matrix<gf<2, 0x7> > mat_out);
    */
};
}

#endif /* LINEARCODEUTILS_H_ */
