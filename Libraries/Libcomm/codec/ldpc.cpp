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
 * ldpc.cpp
 *
 *  Created on: 9 Jul 2009
 *      Author: swesemeyer
 */

#include "ldpc.h"
#include "linear_code_utils.h"
#include "randgen.h"
#include "sumprodalg/spa_factory.h"
#include <cmath>
#include <sstream>
#include <cstdlib>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class GF_q, class real> ldpc<GF_q, real>::ldpc(
      libbase::matrix<GF_q> paritycheck_mat, const int num_of_iters)
   {
   //initialise the provided values;
   this->pchk_matrix = paritycheck_mat;
   this->max_iter = num_of_iters;

   //compute some values from the parity check matrix
   this->length_n = paritycheck_mat.size().cols();
   this->dim_pchk = paritycheck_mat.size().rows();

   //work out the col and row weights (including their maximums)
   //also determine M_n(positions of non-zero entries per column)
   //and N_m (positions of non-zero entries per row)
   this->max_col_weight = 0;
   this->max_row_weight = 0;

   this->col_weight.init(this->length_n);
   this->row_weight.init(this->dim_pchk);

   array1i_t tmp_non_zero_pos;
   tmp_non_zero_pos.init(this->dim_pchk);

   int tmp_weight;
   //do cols first
   this->M_n.init(this->length_n);

   for (int col_indx = 0; col_indx < this->length_n; col_indx++)
      {
      // reset tmp vars
      tmp_weight = 0;
      tmp_non_zero_pos *= 0;

      for (int row_indx = 0; row_indx < this->dim_pchk; row_indx++)
         {
         if (paritycheck_mat(row_indx, col_indx) != GF_q(0))
            {
            tmp_non_zero_pos(tmp_weight) = row_indx + 1;//we count from 1
            tmp_weight++;
            }
         }
      this->col_weight(col_indx) = tmp_weight;
      if (tmp_weight > this->max_col_weight)
         {
         this->max_col_weight = tmp_weight;
         }
      //now store the non-zero positions in M_n
      this->M_n(col_indx).init(tmp_weight);
      for (int i = 0; i < tmp_weight; i++)
         {
         this->M_n(col_indx)(i) = tmp_non_zero_pos(i);
         }
      }

   //do rows next
   tmp_non_zero_pos.init(this->length_n);
   this->N_m.init(this->dim_pchk);

   for (int row_indx = 0; row_indx < this->dim_pchk; row_indx++)
      {
      // reset tmp vars
      tmp_weight = 0;
      tmp_non_zero_pos *= 0;
      for (int col_indx = 0; col_indx < this->length_n; col_indx++)
         {
         if (paritycheck_mat(row_indx, col_indx) != GF_q(0))
            {
            tmp_non_zero_pos(tmp_weight) = col_indx + 1;//we count from 1
            tmp_weight++;
            }
         }
      this->row_weight(row_indx) = tmp_weight;
      if (tmp_weight > this->max_row_weight)
         {
         this->max_row_weight = tmp_weight;
         }
      //now store the non-zero positions in N_m
      this->N_m(row_indx).init(tmp_weight);
      for (int i = 0; i < tmp_weight; i++)
         {
         this->N_m(row_indx)(i) = tmp_non_zero_pos(i);
         }
      }

   //use sensible default values for the rest
   std::string spa_type = "gdl";
   this->spa_alg = libcomm::spa_factory<GF_q, real>::get_spa(spa_type,
         this->length_n, this->dim_pchk, this->M_n, this->N_m,
         this->pchk_matrix);

   std::string clipping_type = "zero";
   real almost_zero = real(1E-100);
   this->spa_alg->set_clipping(clipping_type, almost_zero);

   this->reduce_to_ref = false;
   this->rand_prov_values = "provided";
   this->decodingSuccess = false;
   //we are done and can call init now.
   this->init();
   }

template <class GF_q, class real> void ldpc<GF_q, real>::init()
   {

   //compute the generator matrix for the code

   libbase::linear_code_utils<GF_q>::compute_dual_code(this->pchk_matrix,
         this->gen_matrix, this->perm_to_systematic);

   this->dim_k = this->gen_matrix.size().rows();
   this->info_symb_pos.init(this->dim_k);

   if (this->reduce_to_ref == false)
      {
      //as we define the LDPC code by its parity check matrix,H , the generator matrix
      //will be of the form (P|I) provided H was in systematic form when reduced to REF
      //if H wasn't then perm_to_systematic contains the permutation that transformed
      //H into systematic form which gives us the information we need to extract the
      //positions of the info symbols in G. In fact the last k values of perm_to_systematic
      //are those positions.
      for (int loop = 0; loop < this->dim_k; loop++)
         {
         this ->info_symb_pos(loop) = this->perm_to_systematic((this->length_n
               - this->dim_k) + loop);
         }
      }
   else
      {
      //we reduce the generator matrix to REF format in the hope that the info symbols will be
      //in the first k positions and that we'll therefore have a systematic code
      this->gen_matrix = this->gen_matrix.reduce_to_ref();
      //we now need to find the pivots
      int posy = 0;
      for (int loop = 0; loop < this->dim_k; loop++)
         {
         while (this->gen_matrix(loop, posy) == GF_q(0))
            {
            posy++;
            }
         this->info_symb_pos(loop) = posy;
         }
      }
   }

template <class GF_q, class real> void ldpc<GF_q, real>::do_init_decoder(
      const array1vdbl_t& ptable)
   {

   this->current_iteration = 0;
   this->decodingSuccess = false;

#if DEBUG>=2
   libbase::trace << std::endl << "The first 5 received likelihoods are:" << std::endl;
   libbase::trace<< ptable.extract(0,5);
#endif
   this->received_probs.init(this->length_n);

   //cast the values from double to real
   int numOfElements = GF_q::elements();
   for (int loop_n = 0; loop_n < this->length_n; loop_n++)
      {
      this->received_probs(loop_n).init(numOfElements);
      for (int loop_e = 0; loop_e < numOfElements; loop_e++)
         {
         this->received_probs(loop_n)(loop_e) = real(ptable(loop_n)(loop_e));
         }
      }

   //determine the most likely symbol
   hd_functor(this->received_probs, this->received_word_hd);

#if DEBUG>=2
   libbase::trace << std::endl << "Currently, the most likely received word is:" << std::endl;
   this->received_word_hd.serialize(libbase::trace, ' ');
#endif
   //do not check whether we have a solution already. This will force
   //the algorithm to do at least 1 iteration. That should be enough
   //for the computation of the extrinsic information to work properly.
   //If we stop here then we return the same information back which
   //will result in the extrinsic info to equal 1.

   /*
    //this->isCodeword();
    if (this->decodingSuccess)
    {
    //copy the values over
    //this->computed_solution = this->received_probs;

    this->computed_solution.init(this->length_n);
    for (int loop_n = 0; loop_n < this->length_n; loop_n++)
    {
    this->computed_solution(loop_n).init(numOfElements);
    for (int loop_e = 0; loop_e < numOfElements; loop_e++)
    {
    this->computed_solution(loop_n)(loop_e)
    = static_cast<double> (this->received_probs(loop_n)(loop_e));
    }
    }
    }
    */

   //only do the rest if we don't have a codeword already
   //else
   //   {
   this->spa_alg->spa_init(this->received_probs);
   //   }
   }
template <class GF_q, class real> void ldpc<GF_q, real>::isCodeword()
   {
   bool dec_success = true;
   int num_of_entries = 0;
   int pos_n = 0;

   GF_q tmp_val = GF_q(0);
   int rows = 0;
   while (dec_success && rows < this->dim_pchk)
      {
      tmp_val = GF_q(0);
      num_of_entries = this->N_m(rows).size();
      for (int loop = 0; loop < num_of_entries; loop++)
         {
         pos_n = this->N_m(rows)(loop) - 1;//we count from zero
         tmp_val += this->pchk_matrix(rows, pos_n) * received_word_hd(pos_n);
         }
      if (tmp_val != GF_q(0))
         {
         //the syndrome is non-zero
         dec_success = false;
         }
      rows++;
      }
   this->decodingSuccess = dec_success;
#if DEBUG>=2
   if (dec_success)
      {
      libbase::trace << "We have a solution" << std::endl;
      }
#endif
   }
template <class GF_q, class real> void ldpc<GF_q, real>::do_encode(
      const libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   libbase::linear_code_utils<GF_q>::encode_cw(this->gen_matrix, source,
         encoded);

#if DEBUG>=2
   this->received_word_hd = encoded;
   this->isCodeword();
   assertalways(this->decodingSuccess);
   //extract the info symbols from the codeword word and compare them to the original
   for (int loop_i = 0; loop_i < this->dim_k; loop_i++)
      {
      assertalways(source(loop_i) == encoded(this->info_symb_pos(loop_i)));
      }
#endif
#if DEBUG>=2
   libbase::trace << "The encoded word is:" << std::endl;
   encoded.serialize(libbase::trace, ' ');
   libbase::trace << std::endl;
#endif
   }

template <class GF_q, class real> void ldpc<GF_q, real>::softdecode(
      array1vdbl_t& ri, array1vdbl_t& ro)
   {
   //update the iteration counter
   this->current_iteration++;
   //init the received sd information vector;
   ri.init(this->dim_k);

   //Only continue if we haven't already computed a solution in a previous iteration
   if (this->decodingSuccess)
      {
      //initialise the output vector to the previously computed solution
      ro = this->computed_solution;
      }
   else
      {
      array1vd_t tmp_ro;
      this->spa_alg->spa_iteration(tmp_ro);

#if DEBUG>=3
      libbase::trace << std::endl << "This is iteration: " << this->current_iteration << std::endl;
      libbase::trace
      << "The newly computed normalised probabilities are given by:" << std::endl;
      ro.serialize(libbase::trace, ' ');
#endif

      //determine the most likely symbol
      hd_functor(tmp_ro, this->received_word_hd);

      //cast the values back from real to double
      int num_of_elements = GF_q::elements();
      ro.init(this->length_n);
      for (int loop_n = 0; loop_n < this->length_n; loop_n++)
         {
         ro(loop_n).init(num_of_elements);
         for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
            {
            ro(loop_n)(loop_e) = static_cast<double> (tmp_ro(loop_n)(loop_e));
            }
         }
      //do we have a solution?
      this->isCodeword();
      if (this->decodingSuccess)
         {
         //store the solution for the next iteration
         this->computed_solution = ro;
         }

#if DEBUG>=2
      libbase::trace << std::endl << "This is iteration: " << this->current_iteration << std::endl;
      libbase::trace << "The most likely received word is now given by:" << std::endl;
      this->received_word_hd.serialize(libbase::trace, ' ');
#endif
      //finished decoding
      }

   //extract the info symbols from the received word
   for (int loop_i = 0; loop_i < this->dim_k; loop_i++)
      {
      ri(loop_i) = ro(this->info_symb_pos(loop_i));
      }
#if DEBUG>=3
   libbase::trace << std::endl << "This is iteration: " << this->current_iteration << std::endl;
   libbase::trace << std::endl << "The info symbol probabilities are given by:" << std::endl;
   ri.serialize(libbase::trace, ' ');
#endif

   }

template <class GF_q, class real> std::string ldpc<GF_q, real>::description() const
   {
   std::ostringstream sout;
   sout << "LDPC(n=" << this->length_n << ", m=" << this->dim_pchk << ", k="
         << this->dim_k << ", spa=" << this->spa_alg->spa_type() << ", iter="
         << this->max_iter << ", clipping="
         << this->spa_alg->get_clipping_type() << ", almostzero="
         << this->spa_alg->get_almostzero() << ")";
#if DEBUG>=2
   this->serialize(libbase::trace);
   libbase::trace << std::endl;
#endif

#if DEBUG>=2
   libbase::trace << "Its parity check matrix is given by:" << std::endl;
   this->pchk_matrix.serialize(libbase::trace, '\n');

   libbase::trace << "Its generator matrix is given by:" << std::endl;
   this->gen_matrix.serialize(libbase::trace, '\n');
   libbase::trace << "The information symbols are located in columns:" << std::endl;
   for (int loop = 0; loop < this->dim_k; loop++)
      {
      libbase::trace << this->info_symb_pos(loop) + 1 << " ";
      }
   libbase::trace << std::endl;
#endif
   return sout.str();
   }

/* object serialization - writing
 *
 * As an example, given the matrix (5,3) over GF(4):
 *
 * 2 1 0 0 1
 * 0 2 2 3 3
 * 3 0 2 1 0
 *
 * An example file would be
 * ldpc<gf2,double>
 * #version
 * 4
 * #SPA
 * trad
 * #iter
 * 10
 * #clipping method and almost_zero value
 * zero
 * 1e-100
 * # reduce generator matrix to REF? (true|false)
 * false
 * # length dim
 * 5 3
 * # max col/row weight
 * 2 2
 * #non-zero vals
 * provided
 * # col weights
 * 5
 * 2 2 2 2 2
 * # row weights
 * 3
 * 3 4 3
 * #non-zero pos in cols
 * 2
 * 1 3
 * 2
 * 1 2
 * 2
 * 2 3
 * 2
 * 2 3
 * 2
 * 1 2
 * #non-zero vals in cols
 * 2
 * 2 3
 * 2
 * 1 2
 * 2
 * 2 2
 * 2
 * 3 1
 * 2
 * 1 3
 *
 */
template <class GF_q, class real> std::ostream& ldpc<GF_q, real>::serialize(
      std::ostream& sout) const
   {
   assertalways(sout.good());
   sout << "# Version" << std::endl;
   sout << 5 << std::endl;
   sout << "# SPA type (trad|gdl)" << std::endl;
   sout << this->spa_alg->spa_type() << std::endl;
   sout << "# Number of iterations" << std::endl;
   sout << this->max_iter << std::endl;
   sout << "# Clipping method (zero=replace only zeros, clip=replace values below almostzero)" << std::endl;
   sout << this->spa_alg->get_clipping_type() << std::endl;
   sout << "# Value of almostzero" << std::endl;
   sout << this->spa_alg->get_almostzero() << std::endl;
   sout << "# Reduce generator matrix to REF? (true|false)" << std::endl;
   sout << this->reduce_to_ref << std::endl;
   sout << "# Length (n)" << std::endl;
   sout << this->length_n << std::endl;
   sout << "# Dimension (m)" << std::endl;
   sout << this->dim_pchk << std::endl;
   sout << "# Max column weight" << std::endl;
   sout << this->max_col_weight << std::endl;
   sout << "# Max row weight" << std::endl;
   sout << this->max_row_weight << std::endl;
   sout << "# Non-zero values (ones|random|provided)" << std::endl;
   sout << this->rand_prov_values << std::endl;
   if ("random" == this->rand_prov_values)
      {
      sout << "# Seed for random generator" << std::endl;
      sout << this->seed << std::endl;
      }

   sout << "# Column weight vector" << std::endl;
   sout << this->col_weight;
   sout << "# Row weight vector" << std::endl;
   sout << this->row_weight;

   sout << "# Non zero positions per col" << std::endl;
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      sout << this->M_n(loop1);
   // only output non-zero entries if needed
   if ("provided" == this->rand_prov_values)
      {
      libbase::vector<GF_q> non_zero_vals_in_col;
      sout << "# Non zero values per col" << std::endl;
      for (int loop1 = 0; loop1 < this->length_n; loop1++)
         {
         int num_of_non_zeros = this->M_n(loop1).size();
         non_zero_vals_in_col.init(num_of_non_zeros);
         for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
            {
            int tmp_pos = this->M_n(loop1)(loop2) - 1;
            int gf_val_int = this->pchk_matrix(tmp_pos, loop1);
            assert(gf_val_int != GF_q(0));
            non_zero_vals_in_col(loop2) = gf_val_int;
            }
         sout << non_zero_vals_in_col;
         }
      }
   return sout;
   }

/* object serialization - loading
 *
 * For an example, see the writing method
 */

template <class GF_q, class real> std::istream& ldpc<GF_q, real>::serialize(
      std::istream& sin)
   {
   assertalways(sin.good());
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   assertalways(version>=2);
   std::string spa_type;
   sin >> libbase::eatcomments >> spa_type >> libbase::verify;
   sin >> libbase::eatcomments >> this->max_iter >> libbase::verify;
   assertalways(this->max_iter>=1);
   // Default clipping settings for files with versions less than 3
   std::string clipping_type = "zero";
   real almost_zero = real(1E-100);
   if (version >= 3)
      {
      /* My method of avoiding probs of zero is labelled "zero", while
       * the method of clipping all probs below a certain value is "clip".
       * In either case we need to replace a value by almostzero.
       */
      sin >> libbase::eatcomments >> clipping_type >> libbase::verify;
      assertalways(("clip"==clipping_type)||("zero"==clipping_type));
      double tmp_az;
      sin >> libbase::eatcomments >> tmp_az >> libbase::verify;
      almost_zero = real(tmp_az);
      }
   // Default flag for files with versions less than 4
   this->reduce_to_ref = false;
   if (version >= 5)
      sin >> libbase::eatcomments >> this->reduce_to_ref >> libbase::verify;
   else if (version >= 4)
      {
      std::string tmp_flag;
      sin >> libbase::eatcomments >> tmp_flag >> libbase::verify;
      assertalways(("true"==tmp_flag)||("false"==tmp_flag));
      if ("true" == tmp_flag)
         this->reduce_to_ref = true;
      }
   sin >> libbase::eatcomments >> this->length_n >> libbase::verify;
   sin >> libbase::eatcomments >> this->dim_pchk >> libbase::verify;

   sin >> libbase::eatcomments >> this->max_col_weight >> libbase::verify;
   sin >> libbase::eatcomments >> this->max_row_weight >> libbase::verify;

   libbase::randgen rng;
   //are the non-zero values provided or do we randomly generate them?
   sin >> libbase::eatcomments >> this->rand_prov_values >> libbase::verify;
   assertalways(("ones"==this->rand_prov_values) ||
         ("random"==this->rand_prov_values) ||
         ("provided"==this->rand_prov_values));
   if ("random" == this->rand_prov_values)
      {
      //read the seed value;
      sin >> libbase::eatcomments >> this-> seed >> libbase::verify;
      assertalways(this->seed>=0);
      rng.seed(this->seed);
      }
   //read the col weights and ensure they are sensible
   this->col_weight.init(this->length_n);
   sin >> libbase::eatcomments >> this->col_weight >> libbase::verify;
   assertalways((1<=this->col_weight.min())&&(this->col_weight.max()<=this->max_col_weight));

   //read the row weights and ensure they are sensible
   this->row_weight.init(this->dim_pchk);
   sin >> libbase::eatcomments >> this->row_weight >> libbase::verify;
   assertalways((0<this->row_weight.min())&&(this->row_weight.max()<=this->max_row_weight));

   this->M_n.init(this->length_n);

   //read the non-zero entries pos per col
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      this->M_n(loop1).init(this->col_weight(loop1));
      sin >> libbase::eatcomments >> this ->M_n(loop1) >> libbase::verify;
      //ensure that the number of non-zero pos matches the previously read value
      assertalways(this->M_n(loop1).size()==this->col_weight(loop1));
      }

   //init the parity check matrix and read in the non-zero entries
   this->pchk_matrix.init(this->dim_pchk, this->length_n);
   this->pchk_matrix = GF_q(0);
   libbase::vector<GF_q> non_zero_vals;
   const int num_of_non_zero_elements = GF_q::elements() - 1;
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      const int tmp_entries = this->M_n(loop1).size();
      non_zero_vals.init(tmp_entries);
      if ("ones" == this->rand_prov_values)
         {
         //in the binary case the non-zero values are 1
         non_zero_vals = GF_q(1);
         }
      else if ("random" == this->rand_prov_values)
         {
         for (int loop_e = 0; loop_e < tmp_entries; loop_e++)
            {
            non_zero_vals(loop_e) = GF_q(1 + int(rng.ival(
                  num_of_non_zero_elements)));
            }
         assertalways(non_zero_vals.min()!=GF_q(0));
         }
      else
         {
         sin >> libbase::eatcomments >> non_zero_vals >> libbase::verify;
         assertalways(non_zero_vals.min()!=GF_q(0));
         }
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         const int tmp_pos = this->M_n(loop1)(loop2) - 1;//we count from 0
         this->pchk_matrix(tmp_pos, loop1) = non_zero_vals(loop2);
         }
      }

   //derive the non-zero position per row from the parity check matrix
   this->N_m.init(this->dim_pchk);
   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      const int tmp_entries = this->row_weight(loop1);
      this->N_m(loop1).init(tmp_entries);
      int tmp_pos = 0;
      for (int loop2 = 0; loop2 < this->length_n; loop2++)
         {
         if (GF_q(0) != this->pchk_matrix(loop1, loop2))
            {
            assertalways(tmp_pos<tmp_entries);
            this->N_m(loop1)(tmp_pos) = loop2 + 1;//we count from 0;
            tmp_pos++;
            }
         }
      //tmp_pos should now correspond to the given row weight
      assertalways(tmp_pos==this->row_weight(loop1));
      }
   this->spa_alg = libcomm::spa_factory<GF_q, real>::get_spa(spa_type,
         this->length_n, this->dim_pchk, this->M_n, this->N_m,
         this->pchk_matrix);
   this->spa_alg->set_clipping(clipping_type, almost_zero);
   this->init();
   return sin;
   }

/*
 * This method outputs the alist format of this code as
 * described by MacKay @
 * http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
 *
 * n m q
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col followed by their values (for GF(p>2))
 * pos of each non-zero entry per row followed by their values (for GF(p>2))
 *
 * where
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - q is only provided in non-binary cases where q=|GF(q)|
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 * Note that the last set of positions and values are only used to
 * verify the information provided by the first set.
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway. An example row with 4 non-zero entries would look like this (assuming gf<3,0xB>
 * 1 1 3 3 9 7 10 2
 * ie the non-zero entries at pos 1,3,9 and 10 are 1,3,7 and 2 respectively
 *
 * Similarly a column with 3 non-zero entries over gf<3,0xB> would look like:
 * 3 4 6 3 12 6
 * ie the non-zero entries at pos 3,6 and 12 are 4, 3 and 6 respectively
 *
 * in the binary case the above row and column would simply be given by
 * 3 6 12
 * 1 3 9 10
 *
 * Also note that the alist format expects cols/rows with weight less than
 * the max col/row weight to be padded with extra 0s, eg
 * if a code has max col weight of 5 and a given col only has weight 3 then
 * it would look like
 * 1 4 5 0 0 (each entry immediately followed by the non-zero values in case of non-binary code)
 * These additional 0s need to be ignored
 *
 */

template <class GF_q, class real> std::ostream& ldpc<GF_q, real>::write_alist(
      std::ostream& sout) const
   {
   assertalways(sout.good());
   int numOfElements = GF_q::elements();
   bool nonbinary = (numOfElements > 2);

   // alist format version
   sout << this->length_n << " " << this->dim_pchk;
   if (nonbinary)
      {
      sout << " " << numOfElements;
      }
   sout << std::endl;
   sout << this->max_col_weight << " " << this->max_row_weight << std::endl;
   this->col_weight.serialize(sout, ' ');
   this->row_weight.serialize(sout, ' ');
   int num_of_non_zeros;
   int gf_val_int;
   int tmp_pos;

   //positions per column (and the non-zero values associated with them in the non-binary case)
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      num_of_non_zeros = this->M_n(loop1).size().length();
      for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
         {
         sout << this->M_n(loop1)(loop2) << " ";
         tmp_pos = this->M_n(loop1)(loop2) - 1;
         if (nonbinary)
            {
            gf_val_int = this->pchk_matrix(tmp_pos, loop1);
            sout << gf_val_int << " ";
            }
         }
      //add 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_col_weight - num_of_non_zeros); loop2++)
         {
         sout << "0 ";
         if (nonbinary)
            {
            sout << "0 ";
            }
         }
      sout << std::endl;
      }

   //positions per row (and the non-zero values associated with them in the non-binary case)
   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      num_of_non_zeros = this->N_m(loop1).size().length();
      for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
         {
         sout << this->N_m(loop1)(loop2) << " ";
         tmp_pos = this->N_m(loop1)(loop2) - 1;
         if (nonbinary)
            {
            gf_val_int = this->pchk_matrix(loop1, tmp_pos);
            sout << gf_val_int << " ";
            }
         }
      //add 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_row_weight - num_of_non_zeros); loop2++)
         {
         sout << "0 ";
         if (nonbinary)
            {
            sout << "0 ";
            }
         }
      sout << std::endl;
      }

   return sout;
   }

/* loading of the  alist format of an LDPC code
 * This method expects the following format
 *
 * n m q
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col followed by their values
 * pos of each non-zero entry per row followed by their values
 *
 * where
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - q is only provided in non-binary cases where q=|GF(q)|
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 * Note that the last set of positions and values are only used to
 * verify the information provided by the first set.
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway. An example row with 4 non-zero entries would look like this (assuming gf<3,0xB>
 * 1 1 3 3 9 7 10 2
 * ie the non-zero entries at pos 1,3,9 and 10 are 1,3,7 and 2 respectively
 *
 * Similarly a column with 3 non-zero entries over gf<3,0xB> would look like:
 * 3 4 6 3 12 6
 * ie the non-zero entries at pos 3,6 and 12 are 4, 3 and 6 respectively
 *
 * in the binary case the above row and column would simply be given by
 * 3 6 12
 * 1 3 9 10
 *
 * Also note that the alist format expects cols/rows with weight less than
 * the max col/row weight to be padded with extra 0s, eg
 * if a code has max col weight of 5 and a given col only has weight 3 then
 * it would look like
 * 1 4 5 0 0 (each entry immediately followed by the non-zero values in case of non-binary code)
 * These additional 0s need to be ignored
 *
 */

template <class GF_q, class real> std::istream& ldpc<GF_q, real>::read_alist(
      std::istream& sin)
   {
   assertalways(sin.good());
   int numOfElements = GF_q::elements();
   bool nonbinary = (numOfElements > 2);

   sin >> libbase::eatcomments >> this->length_n >> libbase::verify;
   sin >> libbase::eatcomments >> this->dim_pchk >> libbase::verify;
   if (nonbinary)
      {
      int q;
      sin >> libbase::eatcomments >> q >> libbase::verify;
      assertalways(numOfElements==q);
      }

   sin >> libbase::eatcomments >> this->max_col_weight >> libbase::verify;
   sin >> libbase::eatcomments >> this->max_row_weight >> libbase::verify;

   //read the col weights and ensure they are sensible
   int tmp_col_weight;
   this->col_weight.init(this->length_n);
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      sin >> libbase::eatcomments >> tmp_col_weight >> libbase::verify;
      //is it between 1 and max_col_weight?
      assertalways((1<=tmp_col_weight)&&(tmp_col_weight<=this->max_col_weight));
      this ->col_weight(loop1) = tmp_col_weight;
      }

   //read the row weights and ensure they are sensible
   int tmp_row_weight;
   this->row_weight.init(this->dim_pchk);
   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      sin >> libbase::eatcomments >> tmp_row_weight >> libbase::verify;
      //is it between 1 and max_row_weight?
      assertalways((1<=tmp_row_weight)&&(tmp_row_weight<=this->max_row_weight));
      this ->row_weight(loop1) = tmp_row_weight;
      }

   this->pchk_matrix.init(this->dim_pchk, this->length_n);
   this->M_n.init(this->length_n);

   //read the non-zero entries of the parity check matrix col by col
   //and ensure they make sense
   int tmp_entries;
   int tmp_pos;
   int tmp_val = 1; //this is the default value for the binary case
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      //read in the non-zero row entries
      tmp_entries = this->col_weight(loop1);
      this->M_n(loop1).init(tmp_entries);
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
         this->M_n(loop1)(loop2) = tmp_pos;
         tmp_pos--;//we start counting at 0 internally
         assertalways((0<=tmp_pos)&&(tmp_pos<this->dim_pchk));
         // read the non-zero element in the non-binary case
         if (nonbinary)
            {
            sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
            assertalways((0<=tmp_val)&&(tmp_val<numOfElements));
            }
         this->pchk_matrix(tmp_pos, loop1) = GF_q(tmp_val);
         }
      //discard any padded 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_col_weight - tmp_entries); loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
         assertalways(0==tmp_pos);
         if (nonbinary)
            {
            sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
            assertalways((0==tmp_val));
            }
         }
      }

   //read the non-zero entries of the parity check matrix row by row
   this->N_m.init(this->dim_pchk);
   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      tmp_entries = this->row_weight(loop1);
      this->N_m(loop1).init(tmp_entries);
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
         this->N_m(loop1)(loop2) = tmp_pos;
         tmp_pos--;//we start counting at 0 internally
         assertalways((0<=tmp_pos)&&(tmp_pos<this->length_n));
         // read the non-zero element in the non-binary case
         if (nonbinary)
            {
            sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
            assertalways((0<=tmp_val)&&(tmp_val<numOfElements));
            }
         assertalways(GF_q(tmp_val)==this->pchk_matrix(loop1,tmp_pos));
         }
      //discard any padded 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_row_weight - tmp_entries); loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
         assertalways(0==tmp_pos);
         if (nonbinary)
            {
            sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
            assertalways((0==tmp_val));
            }
         }
      }
   //set some default values
   this->max_iter = 100;
   this->reduce_to_ref = false;
   if (GF_q::dimension() == 1)
      {
      this->rand_prov_values = "ones";
      }
   else
      {
      this->rand_prov_values = "provided";
      }
   this->spa_alg = libcomm::spa_factory<GF_q, real>::get_spa("gdl",
         this->length_n, this->dim_pchk, this->M_n, this->N_m,
         this->pchk_matrix);
   this->spa_alg->set_clipping("zero", real(1e-100));
   this->init();
   return sin;
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
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
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
      template class ldpc<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer ldpc<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "codec", \
            "ldpc<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            ldpc<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
