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

#include "linear_code_utils.h"
#include <iostream>
#include <algorithm>
#include "logrealfast.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class GF_q, class real>
void linear_code_utils<GF_q, real>::compute_dual_code(
      const matrix<GF_q> & orgMat, matrix<GF_q> & dualCodeGenMatrix,
      array1i_t & systematic_perm)
   {
   int length_n = orgMat.size().cols();
   int dim_k = orgMat.size().rows();
   int dim_m = length_n - dim_k;
   matrix<GF_q> refOrgMat;
#if DEBUG>=2
   std::cout << "The original matrix is given by:" << std::endl;
   orgMat.serialize(std::cout, '\n');
#endif
   linear_code_utils::compute_row_dim(orgMat, refOrgMat);

   dim_k = refOrgMat.size().rows();
   dim_m = length_n - dim_k;

   // Now we need to check that the columns are systematic
   //if they aren't then we need to perform some column permutation
   //otherwise the permutation is simple the identy map

   systematic_perm.init(length_n);
   for (int loop1 = 0; loop1 < length_n; loop1++)
      {
      //the identity permutation
      systematic_perm(loop1) = loop1;
      }

   bool needsPermutation =
         !(libbase::linear_code_utils<GF_q, real>::is_systematic(refOrgMat));
   if (needsPermutation)
      {
      //matrix needs its columns permuted before it is in systematic form
      //find the pivots
      int col_pos = 0;
      for (int loop1 = 0; loop1 < dim_k; loop1++)
         {

         while ((GF_q(1)) != refOrgMat(loop1, col_pos))
            {
            col_pos++;
            }
         std::swap(systematic_perm(loop1), systematic_perm(col_pos));
         col_pos++;
         }
#if DEBUG>=2
      std::cout << std::endl << "The permutation is given by:" << std::endl;
      systematic_perm.serialize(std::cout, ' ');
#endif
      if (needsPermutation)
         {
         //rejig the matrix
         matrix<GF_q> tmprefMatrix;
         tmprefMatrix.init(dim_k, length_n);
         int col_index;
         for (int loop1 = 0; loop1 < length_n; loop1++)
            {
            col_index = systematic_perm(loop1);
            tmprefMatrix.insertcol(refOrgMat.extractcol(col_index), loop1);
            }
         refOrgMat = tmprefMatrix;
         }
      }
   //the matrix should now be in the right format
#if DEBUG>=2
   std::cout << "After permuting any columns, the matrix is now given by:" << std::endl;
   refOrgMat.serialize(std::cout, '\n');
#endif

   //extract the P part of G'=(I_k|P)
   matrix<GF_q> tmp_p_mat;
   tmp_p_mat.init(dim_k, dim_m);

   for (int loop = dim_k; loop < length_n; loop++)
      {
      tmp_p_mat.insertcol(refOrgMat.extractcol(loop), loop - dim_k);
      }

   //this will hold -P^t=P^t (as we are in char GF(q)=2)
   matrix<GF_q> tmp_p_mat_t;
   tmp_p_mat_t.init(dim_m, length_n - dim_m);
   //now transpose yourself
   tmp_p_mat_t = tmp_p_mat.transpose();
   matrix<GF_q> id_m = matrix<GF_q>::eye(dim_m);

   //Construct the dual code gen matrix which is of the form
   //H=(-P^t|I_m)
   dualCodeGenMatrix.init(dim_m, length_n);

   //insert the transposed matrix
   for (int loop = 0; loop < (length_n - dim_m); loop++)
      {
      dualCodeGenMatrix.insertcol(tmp_p_mat_t.extractcol(loop), loop);
      }

   //now add the identity matrix
   int counter = 0;
   for (int loop = (length_n - dim_m); loop < length_n; loop++)
      {
      dualCodeGenMatrix.insertcol(id_m.extractcol(counter), loop);
      counter++;
      }

#if DEBUG>=2
   std::cout << "The generator matrix of the permuted dual code is given by:" << std::endl;
   dualCodeGenMatrix.serialize(std::cout, '\n');
#endif

   //undo any permutation that we did
   if (needsPermutation)
      {

      matrix<GF_q> tmpdualCodeGenMatrix;
      tmpdualCodeGenMatrix.init(dim_m, length_n);
      int col_index;
      for (int loop1 = 0; loop1 < length_n; loop1++)
         {
         col_index = systematic_perm(loop1);
         tmpdualCodeGenMatrix.insertcol(dualCodeGenMatrix.extractcol(loop1),
               col_index);
         }
      dualCodeGenMatrix = tmpdualCodeGenMatrix;
      }

#if DEBUG>=2
   std::cout
   << "After undoing the permutation, the generator matrix of the dual code is given by:" << std::endl;
   dualCodeGenMatrix.serialize(std::cout, '\n');
#endif

#if DEBUG>=2
   std::cout
   << "we now multiply the generator matrix by the transpose of the original matrix, ie:" << std::endl;
   std::cout << "the transpose of:" << std::endl;
   orgMat.serialize(std::cout, '\n');
   std::cout << "is:" << std::endl;
   matrix<GF_q> trans(orgMat.transpose());
   trans.serialize(std::cout, '\n');
   matrix<GF_q> zeroTest(dualCodeGenMatrix*trans);
   std::cout << "the result is:" << std::endl;
   zeroTest.serialize(std::cout, '\n');
   assertalways(GF_q(0)==zeroTest.max());
#endif

   }

template <class GF_q, class real>
void linear_code_utils<GF_q, real>::compute_row_dim(const matrix<GF_q>& orgMat,
      matrix<GF_q> & maxRowSpaceMat)
   {
   int length_n = orgMat.size().cols();
   int dim_k = orgMat.size().rows();

   //copy original matrix and reduce it to REF, ie G'=(I_k|P)
   matrix<GF_q> refOrgMat(orgMat.reduce_to_ref());

#if DEBUG>=2
   std::cout << "The REF is given by:" << std::endl;
   refOrgMat.serialize(std::cout, '\n');
#endif

   //quick check that we at least have the right dimension
   //since the matrix is in REF, the last rows of the matrix
   //should not be equal to zero. If the last row of the matrix is zero
   //then keep dropping it until it isn't. Consequently, the row dimension
   //of the generator matrix is reduced by the number of dropped rows.
   int loop1 = dim_k;
   int loop2;
   bool isZero = true;
   while (isZero && (loop1 > 0))
      {
      loop1--;
      loop2 = length_n;
      while (isZero && (loop2 >= dim_k))
         {
         loop2--;
         if (refOrgMat(loop1, loop2) != (GF_q(0)))
            {
            isZero = false;
            }
         }
      }
#if DEBUG>=2
   std::cout << "We have " << (dim_k - loop1) - 1 << " zero rows." << std::endl;
#endif
   //compensate for the fact the we start counting rows from 0
   loop1++;
   if (loop1 < dim_k)
      {
      dim_k = loop1;
      //the matrix contains zero rows - drop them

      maxRowSpaceMat.init(dim_k, length_n);
      for (loop2 = 0; loop2 < dim_k; loop2++)
         {
         maxRowSpaceMat.insertrow(refOrgMat.extractrow(loop2), loop2);
         }
#if DEBUG>=2
      std::cout << "After dropping zero rows, the REF is given by:" << std::endl;
      maxRowSpaceMat.serialize(std::cout, '\n');
#endif
      }
   else
      {
      //the original matrix is ok already
      maxRowSpaceMat = refOrgMat;
      }
   }

template <class GF_q, class real>
void linear_code_utils<GF_q, real>::remove_zero_cols(const matrix<GF_q>& mat_G,
      matrix<GF_q> noZeroCols_G)
   {
   //dummy implementations
   //TODO fix me
   noZeroCols_G = mat_G;

   }
template <class GF_q, class real>
void linear_code_utils<GF_q, real>::encode_cw(const matrix<GF_q> & mat_G,
      const array1i_t & source, array1i_t & encoded)
   {
#if DEBUG>=2
   libbase::trace << std::endl << "encoding";
#endif
   //initialise encoded
   int length_n = mat_G.size().cols();
   int dim_k = mat_G.size().rows();

   assertalways(dim_k == source.size().length());

   encoded.init(length_n);
   for (int i = 0; i < length_n; i++)
      {
      GF_q val = GF_q(0);
      for (int j = 0; j < dim_k; j++)
         {
         val += GF_q(source(j)) * mat_G(j, i);
         }
      encoded(i) = val;
      }
#if DEBUG>=2
   libbase::trace << std::endl << "finished encoding";
#endif
   }

template <class GF_q, class real>
bool linear_code_utils<GF_q, real>::compute_syndrome(
      const matrix<GF_q> & parMat, const array1gfq_t & received_word_hd,
      array1gfq_t & syndrome_vec)
   {
   bool dec_success = true;
   int dim_m = parMat.size().rows();
   int length_n = parMat.size().cols();

   //check that we have compatible lengths
   assertalways(received_word_hd.size().length() == length_n);

   syndrome_vec.init(dim_m);
   GF_q tmp_val = GF_q(0);

   for (int rows = 0; rows < dim_m; rows++)
      {
      tmp_val = GF_q(0);
      for (int cols = 0; cols < length_n; cols++)
         {
         tmp_val += parMat(rows, cols) * received_word_hd(cols);
         }
      if (tmp_val != GF_q(0))
         {
         //the syndrome is non-zero
         dec_success = false;
         }
      syndrome_vec(rows) = tmp_val;

      }
   return dec_success;

   }

template <class GF_q, class real>
bool linear_code_utils<GF_q, real>::is_systematic(const matrix<GF_q> & genMat)
   {
   int dim_k = genMat.size().rows();
   int loop1 = 0;

   bool isSystematic = true;
   //easy check - is the diagonal made of 1s
   while (isSystematic && (loop1 < dim_k))
      {
      if (genMat(loop1, loop1) != GF_q(1))
         {
         isSystematic = false;
         }
      loop1++;
      }
   //check that the non-diagonal entries are zero
   loop1 = 0;
   int loop2 = 1;
   while (isSystematic && (loop1 < dim_k))
      {
      do
         {
         if ((GF_q(0) != genMat(loop1, loop2)) && (loop1 != loop2))
            {
            isSystematic = false;
            }
         loop2++;
         } while (isSystematic && (loop2 < dim_k));
      loop2 = 0;
      loop1++;
      }
   return isSystematic;
   }

template <class GF_q, class real>
void linear_code_utils<GF_q, real>::create_hadamard(matrix<int>& hadMat, int m)
   {
   assertalways(m>1);
   int num_of_elements = 1 << m;
   hadMat.init(num_of_elements, num_of_elements);
   matrix<int> A;
   A.init(2, 2);
   A = 1;
   A(1, 1) = -1;
   int size = 2;
   if (num_of_elements == 2)
      {
      std::swap(hadMat, A); //hadMat=A;
      }
   else
      {
      matrix<int> B(A);
      matrix<int> C;
      while (size < num_of_elements)
         {
         linear_code_utils::compute_kronecker(A, B, C);
         std::swap(B, C);
         size = size * 2;
         }
      std::swap(hadMat, B); //B contains the Hadamard matrix
#if DEBUB>=2
            hadMat.serialize(std::cerr,' ');
#endif
      }
   }

template <class GF_q, class real>
void linear_code_utils<GF_q, real>::compute_kronecker(const matrix<int>& A,
      const matrix<int>& B, matrix<int>& prod)
   {
   //ensure the product is big enough
   int row_a = A.size().rows();
   int row_b = B.size().rows();
   int col_a = A.size().cols();
   int col_b = B.size().cols();
   int rowsize = row_a * row_b;
   int colsize = col_a * col_b;
   int pos_m = 0;
   int pos_n = 0;
   prod.init(rowsize, colsize);

   for (int loop_ra = 0; loop_ra < row_a; loop_ra++)
      {
      pos_m = loop_ra * row_b;
      for (int loop_rb = 0; loop_rb < row_b; loop_rb++)
         {
         for (int loop_ca = 0; loop_ca < col_a; loop_ca++)
            {
            pos_n = loop_ca * col_b;

            for (int loop_cb = 0; loop_cb < col_b; loop_cb++)
               {
               prod(pos_m, pos_n) = A(loop_ra, loop_ca) * B(loop_rb, loop_cb);
               pos_n++;
               }
            }
         pos_m++;
         }
      }
   }

} // end namespace

#include "gf.h"
#include "mpreal.h"

namespace libbase {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

#define REAL_TYPE_SEQ \
   (double)(mpreal)

#define INSTANTIATE(r, args) \
      template class linear_code_utils<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
