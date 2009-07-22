/*
 * linear_code_utils.cpp
 *
 *  Created on: 10 Jul 2009
 *      Author: swesemeyer
 */

#include "linear_code_utils.h"
#include <iostream>
#include <algorithm>

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class GF_q> linear_code_utils<GF_q>::linear_code_utils()
   {
   //nothing to do

   }

template <class GF_q> linear_code_utils<GF_q>::~linear_code_utils()
   {
   //nothing to dob
   }

template <class GF_q> void linear_code_utils<GF_q>::compute_dual_code(
      const matrix<GF_q> & orgMat, matrix<GF_q> & dualCodeGenMatrix,
      array1i_t & systematic_perm)
   {
   int length_n = orgMat.size().cols();
   int dim_k = orgMat.size().rows();
   int dim_m = length_n - dim_k;

   //copy original matrix and reduce it to REF, ie G'=(I_k|P)
   matrix<GF_q> tmpOrgMat(orgMat);
   matrix<GF_q> refOrgMat(tmpOrgMat.reduce_to_ref());

#if DEBUG>=2
   std::cout << "The REF is given by:\n";
   refOrgMat.serialize(std::cout, '\n');
#endif

   //quick check that we at least have the right dimension
   //since the matrix is in REF, the last rows of the matrix
   //should not be equal to zero. If it is then the row dimension
   //of the generator matrix needs to be reduced.
   int loop1 = dim_k;
   int loop2;
   bool isZero = true;
   while (isZero && (loop1 >= 0))
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
   std::cout << "We have " << (dim_k - loop1) - 1 << " zero rows.\n";
#endif
   //compensate for the fact the we start counting rows from 0
   loop1++;
   if (loop1 < dim_k)
      {
      dim_k = loop1;
      dim_m = length_n - dim_k;
      //the matrix contains zero rows - drop them
      matrix<GF_q> tmprefMatrix;
      tmprefMatrix.init(dim_k, length_n);
      for (loop2 = 0; loop2 < dim_k; loop2++)
         {
         tmprefMatrix.insertrow(refOrgMat.extractrow(loop2), loop2);
         }
      refOrgMat = tmprefMatrix;
      }

#if DEBUG>=2
   std::cout << "After dropping zero rows, the REF is given by:\n";
   refOrgMat.serialize(std::cout, '\n');
#endif
   // Now we need to check that the columns are systematic
   //if they aren't then we need to perform some column permutation
   //otherwise the permutation is simple the identy map

   systematic_perm.init(length_n);
   for (loop1 = 0; loop1 < length_n; loop1++)
      {
      //the identity permutation
      systematic_perm(loop1) = loop1;
      }

   bool needsPermutation = false;
   bool swapcols = false;
   if (!libbase::linear_code_utils<GF_q>::is_systematic(refOrgMat))
      {
      //matrix needs its columns permuted before it is in systematic form
      int col_pos = 0;
      for (loop1 = 0; loop1 < dim_k; loop1++)
         {

         while ((GF_q(1)) != refOrgMat(loop1, col_pos))
            {
            col_pos++;
            needsPermutation = true;
            swapcols = true;
            }
         if (swapcols)
            {
            systematic_perm(col_pos) = systematic_perm(loop1);
            systematic_perm(loop1) = col_pos;
            swapcols = false;
            }
         col_pos++;
         }
#if DEBUG>=2
      std::cout << "\nThe permutation is given by:\n";
      systematic_perm.serialize(std::cout, ' ');
#endif
      if (needsPermutation)
         {
         //rejig the matrix
         matrix<GF_q> tmprefMatrix;
         tmprefMatrix.init(dim_k, length_n);
         int col_index;
         for (loop1 = 0; loop1 < length_n; loop1++)
            {
            col_index = systematic_perm(loop1);
            tmprefMatrix.insertcol(refOrgMat.extractcol(col_index), loop1);
            }
         refOrgMat = tmprefMatrix;
         }
      }
   //the matrix should now be in the right format
#if DEBUG>=2
   std::cout << "After permuting any columns, the matrix is now given by:\n";
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
   std::cout << "The generator matrix of the permuted dual code is given by:\n";
   dualCodeGenMatrix.serialize(std::cout, '\n');
#endif

   //undo any permutation that we did
   if (needsPermutation)
      {

      matrix<GF_q> tmpdualCodeGenMatrix;
      tmpdualCodeGenMatrix.init(dim_m, length_n);
      int col_index;
      for (loop1 = 0; loop1 < length_n; loop1++)
         {
         col_index = systematic_perm(loop1);
         tmpdualCodeGenMatrix.insertcol(dualCodeGenMatrix.extractcol(loop1),
               col_index);
         }
      dualCodeGenMatrix = tmpdualCodeGenMatrix;
      }

#if DEBUG>=2
   std::cout
   << "After undoing the permutation, the generator matrix of the dual code is given by:\n";
   dualCodeGenMatrix.serialize(std::cout, '\n');
#endif

   }

template <class GF_q> void linear_code_utils<GF_q>::compute_row_dim(
      const matrix<GF_q>& parMat_H, matrix<GF_q> & maxRowSpace_H)
   {
   //dummy implementations
   //TODO fix me
   maxRowSpace_H = parMat_H;
   }

template <class GF_q> void linear_code_utils<GF_q>::remove_zero_cols(
      const matrix<GF_q>& mat_G, matrix<GF_q> noZeroCols_G)
   {
   //dummy implementations
   //TODO fix me
   noZeroCols_G = mat_G;

   }
template <class GF_q> void linear_code_utils<GF_q>::encode_cw(
      const matrix<GF_q> & mat_G, const array1i_t & source, array1i_t & encoded)
   {
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
   }

template <class GF_q> bool linear_code_utils<GF_q>::compute_syndrome(
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

template <class GF_q> bool linear_code_utils<GF_q>::is_systematic(const matrix<
      GF_q> & genMat)
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

//determine the most likely symbol
template <class GF_q> void linear_code_utils<GF_q>::get_most_likely_received_word(
      const array1dv_t& received_likelihoods, array1d_t & received_word_sd,
      array1gfq_t& received_word_hd)
   {
   //some helper variables
   int length_n = received_likelihoods.size();
   double mostlikely_sofar = 0;
   int indx = 0;
   array1d_t tmp_vec;
   int num_of_symbs;

   received_word_sd.init(length_n);
   received_word_hd.init(length_n);

   for (int loop_n = 0; loop_n < length_n; loop_n++)
      {
      mostlikely_sofar = 0;
      indx = 0;
      tmp_vec = received_likelihoods(loop_n);
      num_of_symbs = tmp_vec.size();
      for (int loop_q = 0; loop_q < num_of_symbs; loop_q++)
         {
         if (mostlikely_sofar <= tmp_vec(loop_q))
            {
            mostlikely_sofar = tmp_vec(loop_q);
            indx = loop_q;
            }
         }
      received_word_sd(loop_n) = mostlikely_sofar;
      received_word_hd(loop_n) = GF_q(indx);
      }
   }
}
//explicit realisations
namespace libbase {
template class linear_code_utils<gf<1, 0x3> > ;
template class linear_code_utils<gf<2, 0x7> > ;
template class linear_code_utils<gf<3, 0xB> > ;
template class linear_code_utils<gf<4, 0x13> > ;
template class linear_code_utils<gf<5, 0x25> > ;
template class linear_code_utils<gf<6, 0x43> > ;
template class linear_code_utils<gf<7, 0x89> > ;
}
