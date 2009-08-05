/*
 * LinearGFqCodeUtils.h
 *
 *  Created on: 10 Jul 2009
 *      Author: swesemeyer
 */

#ifndef LINEARCODEUTILS_H_
#define LINEARCODEUTILS_H_

#include "matrix.h"
#include "gf.h"
using libbase::matrix;

namespace libbase {
/*!
 *  \brief   Linear Block Code Helper Class
 *  \author  S Wesemeyer
 * This class is a helper class and provides utility methods for
 * linear block codes over GF(2^p) with p>=1
 *
 * \section svn Version Control
 * - $Revision: 2668 $
 * - $Date: 2009-07-16 15:41:52 +0100 (Thu, 16 Jul 2009) $
 * - $Author: swesemeyer $
 */

template <class GF_q, class real = double>
class linear_code_utils {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<GF_q> array1gfq_t;
   typedef libbase::vector<array1d_t> array1dv_t;

   linear_code_utils();
   virtual ~linear_code_utils();

   /*!This computes the dual code of the given matrix as follows:
    * 1) reduce orgMat to REF, call this matrix G
    * 2) Let G'=\pi(G) so that it is in the following format G'=(I|P)
    * 3) Let H'=(-P^t|I)
    * 4) Let H=\pi^{-1}(H')
    * H is then the generator matrix of the dual code of G. H is also
    * the parity check matrix of G.
    */
   static void compute_dual_code(const matrix<GF_q> & orgMat,
         matrix<GF_q> & dualCodeMatrix, array1i_t & systematic_perm);

   /*
    * !This computes the row space of a parity check matrix,
    * ie given an mxn matrix H, this method determines the linear
    * independent rows of H and returns a new matrix consisting only
    * of those rows.
    * Note this is mainly useful for LDPC codes that are defined by their
    * parity check matrix.
    */
   static void compute_row_dim(const matrix<GF_q> & parMat_H,
         matrix<GF_q> & maxRowSpace_H);

   /*
    * !This removes any zero columns from a matrix and returns a new matrix
    * without these columns.
    */
   static void remove_zero_cols(const matrix<GF_q> & mat_G,
         matrix<GF_q> noZeroCols_G);

   /*
    * !This encodes a codeword given a generator matrix
    */
   static void encode_cw(const matrix<GF_q> & mat_G, const array1i_t & source,
         array1i_t & encoded);

   /*
    * !This computes the syndrome of a received word - it returns true if the syndrome is the zero vector.
    *
    */
   static bool compute_syndrome(const matrix<GF_q> & parMat,
         const array1gfq_t & received_word_hd, array1gfq_t & syndrome);

   /*! This checks whether the generator matrix is in systematic form (eg G=(I|P))
    *
    */
   static bool is_systematic(const matrix<GF_q> & genMat);

   /*!
    * This simply extracts the most likely received word in soft decision format
    */
   static void get_most_likely_received_word(
         const array1dv_t& received_likelihoods, array1d_t & received_word_sd,
         array1gfq_t & received_word_hd);

   /*!
    * This creates a Hadamard matrix of size=2^m
    */
   static void create_hadamard(matrix<int>& hadMat, int m);

   /*!
    * This computes the Kroenecker product of 2 matrices
    * if A is an mxn matrix and B is a pxq matrix then their Kroenecker product is
    * an (m*p)(n*q) matrix. The result of the product is that each a_i_j in A is effectively
    * replaced by the matrix a_i_j*B.
    */
   static void compute_kronecker(const matrix<int>& A, const matrix<int>& B,
         matrix<int>& prod);

   /*! \brief convert a parity check matrix over GF(q) into the equivalent binary one
    *
    */
   /*
    static void convert_to_binary(const matrix<GF_q>& mat_in,
    matrix<gf<2, 0x7> > mat_out);
    */
};
}

#endif /* LINEARCODEUTILS_H_ */
