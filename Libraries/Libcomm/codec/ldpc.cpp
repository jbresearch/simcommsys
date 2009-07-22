/*
 * ldpc.cpp
 *
 *  Created on: 9 Jul 2009
 *      Author: swesemeyer
 */

#include "ldpc.h"
#include "linear_code_utils.h"
#include <math.h>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class GF_q> void ldpc<GF_q>::init()
   {

   //compute the generator matrix for the code

   libbase::linear_code_utils<GF_q>::compute_dual_code(this->pchk_matrix,
         this->gen_matrix, this->perm_to_systematic);

   //as we define the LDPC code by its parity check matrix,H , the generator matrix
   //will be of the form (P|I) provided H was in systematic form when reduced to REF
   //if H wasn't then perm_to_systematic contains the permutation that transformed
   //H into systematic form which gives us the information we need to extract the
   //positions of the info symbols in G. In fact the last k values of perm_to_systematic
   //are those positions.

   this->dim_k = this->gen_matrix.size().rows();
   this->info_symb_pos.init(this->dim_k);
   for (int loop = 0; loop < this->dim_k; loop++)
      {
      this ->info_symb_pos(loop) = this->perm_to_systematic((this->length_n
            - this->dim_k) + loop);
      }

   //allocate enough storage for the q_mxn and r_mxn values.
   this->marginal_probs.init(this->dim_pchk, this->length_n);

   }

template <class GF_q> void ldpc<GF_q>::resetpriors()
   {
   //not needed
   }

template <class GF_q> void ldpc<GF_q>::setpriors(const array1vd_t & ptable)
   {
   failwith("Not implemented as this function is not needed");

   }

template <class GF_q> void ldpc<GF_q>::setreceiver(const array1vd_t& ptable)
   {

   this->current_iteration = 0;

#if DEBUG>=2
   libbase::trace << "\nThe received likelihoods are:\n";
   ptable.serialize(libbase::trace, ' ');
#endif

   this->received_probs.init(this->length_n);

   //initialise the computed probabilities by normalising the received likelihoods,
   //eg in the binary case compute
   //P(x_n=0)=P(y_n|x_n=0)/(P(y_n|x_n=0)+P(y|x_n=1))
   //P(x_n=1)=P(y_n|x_n=1)/(P(y_n|x_n=0)+P(y|x_n=1))
   //generalise this formula to all elements in GF_q
   //The result of this is that the computed probs for each entry adds to 1
   //and the most likely symbol has the highest probability
   double alpha = 0.0;
   int num_of_elements = GF_q::elements();
   for (int loop_n = 0; loop_n < this->length_n; loop_n++)
      {
      alpha = ptable(loop_n).sum(); //sum all the likelihoods in ptable(loop_n)
      this->received_probs(loop_n) = ptable(loop_n) / alpha; //normalise them
      }

#if DEBUG>=2
   libbase::trace << "\nThe normalised likelihoods are given by:\n";
   this->computed_probs.serialize(libbase::trace, ' ');
#endif

   //determine the most likely symbol
   libbase::linear_code_utils<GF_q>::get_most_likely_received_word(
         this->received_probs, this->received_word_sd, this->received_word_hd);

#if DEBUG>=2
   libbase::trace << "\nCurrently, the most likely received word is:\n";
   this->received_word_hd.serialize(libbase::trace, ' ');
   libbase::trace << "\n the symbol probabilities are given by:\n";
   this->received_word_sd.serialize(libbase::trace, ' ');
#endif
   //do we have a solution already?
   this->decodingSuccess = libbase::linear_code_utils<GF_q>::compute_syndrome(
         this->pchk_matrix, this->received_word_hd, this->syndrome);

#if DEBUG>=2
   libbase::trace << "\nThe syndrome is given by:\n";
   this->syndrome.serialize(libbase::trace, ' ');
#endif

   //only do the rest if we don't have a codeword already
   if (this->decodingSuccess)
      {
      //initialise the output vector to the previously computed solution
      this->computed_solution = this->received_probs;
      }
   else

      {
      //initialise the marginal prob values

      //this uses the description of the algorithm as given by
      //MacKay in Information Theory, Inference and Learning Algorithms(2003)
      //on page 560 - chapter 47.3

      //some helper variables
      int pos = 0;
      int non_zeros = 0;

      //simply set q_mxn(0)=P_n(0)=P(x_n=0) and q_mxn(1)=P_n(1)=P(x_n=1)
      for (int loop_m = 0; loop_m < this->dim_pchk; loop_m++)
         {
         non_zeros = this->N_m(loop_m).size();
         for (int loop_n = 0; loop_n < non_zeros; loop_n++)
            {
            pos = this->N_m(loop_m)(loop_n) - 1;//we count from zero;
            this->marginal_probs(loop_m, pos).q_mxn = this->received_probs(pos);
            this->marginal_probs(loop_m, pos).r_mxn.init(num_of_elements);
            this->marginal_probs(loop_m, pos).r_mxn = 0.0;
            }
         }

#if DEBUG>=2
      libbase::trace << "LDPC Memory Usage:\n ";
      libbase::trace << this->marginal_probs.size()
      * sizeof(ldpc<GF_q>::marginals) / double(1 << 20) << " MB\n";

      libbase::trace << "\nThe marginal matrix is given by:\n";
      this->print_marginal_probs(libbase::trace);
#endif
      }
   }

template <class GF_q> void ldpc<GF_q>::encode(
      const libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   libbase::linear_code_utils<GF_q>::encode_cw(this->gen_matrix, source,
         encoded);
#if DEBUG>=2
   libbase::trace << "The encoded word is:\n";
   encoded.serialize(libbase::trace, ' ');
   libbase::trace << "\n";
#endif
   }

template <class GF_q> void ldpc<GF_q>::softdecode(array1vd_t& ri)
   {
   //this simply calls the proper decoder
   array1vd_t ro;
   this->softdecode(ri, ro);
   }

template <class GF_q> void ldpc<GF_q>::softdecode(array1vd_t& ri,
      libbase::vector<array1d_t>& ro)
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
      for (int loop_m = 0; loop_m < this->dim_pchk; loop_m++)
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
      libbase::trace << "\nThis is iteration: " << this->current_iteration << "\n";
      libbase::trace
      << "After the horizontal step, the marginal matrix is given by:\n";
      this->print_marginal_probs(libbase::trace);
#endif

      //this array holds the checks that use symbol n
      array1i_t M_n;
      //the number of checks in that array
      int size_M_n;

      //loop over all the bit nodes - the vertical step

      for (int loop_n = 0; loop_n < this->dim_pchk; loop_n++)
         {
         M_n = this->M_n(loop_n);
         size_M_n = M_n.size().length();
         for (int loop_m = 0; loop_m < size_M_n; loop_m++)
            {
            this->compute_q_mn(loop_m, loop_n, M_n);
            }
         }
#if DEBUG>=2
      libbase::trace << "\nThis is iteration: " << this->current_iteration << "\n";
      libbase::trace
      << "After the vertical step, the marginal matrix is given by:\n";
      this->print_marginal_probs(libbase::trace);
#endif

      //compute the new probabilities for all symbols given the information in this iteration.
      //This will be used in a tentative decoding to see whether we have found a codeword
      this->compute_probs(ro);

#if DEBUG>=3
      libbase::trace << "\nThis is iteration: " << this->current_iteration << "\n";
      libbase::trace
      << "The newly computed normalised probabilities are given by:\n";
      ro.serialize(libbase::trace, ' ');
#endif

      //determine the most likely symbol
      libbase::linear_code_utils<GF_q>::get_most_likely_received_word(ro,
            this->received_word_sd, this->received_word_hd);

      //do we have a solution?
      this->decodingSuccess
            = libbase::linear_code_utils<GF_q>::compute_syndrome(
                  this->pchk_matrix, this->received_word_hd, this->syndrome);
      if (this->decodingSuccess)
         {
         //store the solution for the next iteration
         this->computed_solution = ro;
         }

#if DEBUG>=2
      libbase::trace << "\nThis is iteration: " << this->current_iteration << "\n";
      libbase::trace << "The most likely received word is now given by:\n";
      this->received_word_hd.serialize(libbase::trace, ' ');
      libbase::trace << "\nIts symbol probabilities are given by:\n";
      this->received_word_sd.serialize(libbase::trace, ' ');
      libbase::trace << "\nIts syndrome is given by:\n";
      this->syndrome.serialize(libbase::trace, ' ');
#endif
      //finished decoding
      }

   //extract the info symbols from the received word
   for (int loop_i = 0; loop_i < this->dim_k; loop_i++)
      {
      ri(loop_i) = ro(this->info_symb_pos(loop_i));
      }
#if DEBUG>=3
   libbase::trace << "\nThis is iteration: " << this->current_iteration << "\n";
   libbase::trace << "\nThe info symbol probabilities are given by:\n";
   ri.serialize(libbase::trace, ' ');
#endif

   }

template <class GF_q> void ldpc<GF_q>::compute_r_mn(int m, int n,
      const array1i_t & tmpN_m)
   {
   //the number of remaining symbols that can vary
   int num_of_var_syms = tmpN_m.size() - 1;
   int num_of_elements = GF_q::elements();
   //for each check node we need to consider num_of_elements^num_of_var_symbols cases
   int num_of_cases = pow(num_of_elements, num_of_var_syms);
   int pos_n = tmpN_m(n) - 1;//we count from 1;
   int bitmask = num_of_elements - 1;

   //only use the entries that are variable
   array1i_t rel_N_m;
   rel_N_m.init(num_of_var_syms);
   int indx = 0;
   for (int loop = 0; loop < num_of_var_syms; loop++)
      {
      if (indx == n)
         {
         indx++;
         }
      rel_N_m(loop) = tmpN_m(indx);
      indx++;
      }
   //go through all cases - this will use bitwise manipulation
   GF_q syndrome_sym = GF_q(0);

   int int_sym_val;
   int bits;
   int pos_n_dash;
   double q_nm_prod = 1.0;

   this->marginal_probs(m, pos_n).r_mxn = 0.0;

   for (int loop1 = 0; loop1 < num_of_cases; loop1++)
      {
      bits = loop1;
      syndrome_sym = GF_q(0);
      q_nm_prod = 1.0;
      for (int loop2 = 0; loop2 < num_of_var_syms; loop2++)
         {

         pos_n_dash = rel_N_m(loop2) - 1;//we count from zero
         //extract int value of the first symbol
         int_sym_val = bits & bitmask;
         //shift bits to the right by the dimension of the finite field
         bits = bits >> GF_q::dimension();

         //add it to the syndrome
         syndrome_sym = syndrome_sym + GF_q(int_sym_val);
         q_nm_prod *= this->marginal_probs(m, pos_n_dash).q_mxn(int_sym_val);
         }
      //adjust the appropriate rmn value
      int_sym_val = syndrome_sym; //since we work in char(GF_q)=2 syndrome_sym=-syndrome_sym
      this->marginal_probs(m, pos_n).r_mxn(int_sym_val) += q_nm_prod;
      }
   }

template <class GF_q> void ldpc<GF_q>::compute_q_mn(int m, int n,
      const array1i_t & M_n)
   {
   //todo avoid the use of temp array
   //initialise some helper variables
   int num_of_elements = GF_q::elements();
   array1d_t q_mn(this -> received_probs(n));

   int m_dash = 0;
   int pos_m = M_n(m) - 1;//we count from 1;

   //compute q_mn(sym) = a_mxn * P_n(sym) * \prod_{m'\in M(n)\m} r_m'xn(0) for all sym in GF_q
   int size_of_M_n = M_n.size().length();
   for (int loop_m = 0; loop_m < size_of_M_n; loop_m++)
      {
      if (m != loop_m)
         {
         m_dash = M_n(loop_m) - 1; //we start counting from zero
         for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
            {
            q_mn(loop_e) *= this->marginal_probs(m_dash, n).r_mxn(loop_e);
            }
         }
      }
   //normalise the q_mxn's so that q_mxn_0+q_mxn_1=1

   double a_nxm = q_mn.sum();//sum up the values in q_mn
   q_mn /= a_nxm; //normalise

   //store the values
   this->marginal_probs(pos_m, n).q_mxn = q_mn;
   }

template <class GF_q> void ldpc<GF_q>::compute_probs(array1vd_t& ro)
   {
   //ensure the output vector has the right length
   ro.init(this->length_n);

   //initialise some helper variables
   int num_of_elements = GF_q::elements();
   double a_n = 0.0;
   int size_of_M_n = 0;
   int pos_m;
   for (int loop_n = 0; loop_n < this->length_n; loop_n++)
      {
      ro(loop_n) = this->received_probs(loop_n);
      size_of_M_n = this->M_n(loop_n).size();
      for (int loop_m = 0; loop_m < size_of_M_n; loop_m++)
         {
         pos_m = this->M_n(loop_n)(loop_m) - 1;//we count from 0
         for (int loop_e = 0; loop_e < num_of_elements; loop_e++)
            {
            ro(loop_n)(loop_e) *= this->marginal_probs(pos_m, loop_n).r_mxn(
                  loop_e);
            }
         }
      //normalise the result so that q_n_0+q_n_1=1
      a_n = ro(loop_n).sum();
      ro(loop_n) /= a_n;
      }
   }

template <class GF_q> void ldpc<GF_q>::print_marginal_probs(std::ostream& sout)
   {
   int num_of_elements = GF_q::elements();
   bool used;
   for (int loop_m = 0; loop_m < this->dim_pchk; loop_m++)
      {
      sout << "\n[";
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
         sout << ")>";
         }
      sout << "]\n";
      }
   }

template <class GF_q> std::string ldpc<GF_q>::description() const
   {
   std::ostringstream sout;
   sout << "LDPC(n=" << this->length_n << ", m=" << this->dim_pchk << ")\n";
   this->serialize(libbase::trace);
   libbase::trace << "\n";
   libbase::trace << "Its parity check matrix is given by:\n";
   this->pchk_matrix.serialize(libbase::trace, '\n');
#if DEBUG>=2
   libbase::trace << "its parity check matrix in REF is given by:\n";
   this->pchk_matrix.reduce_to_ref().serialize(libbase::trace, '\n');
#endif
   libbase::trace << "Its generator matrix is given by:\n";
   this->gen_matrix.serialize(libbase::trace, '\n');
   libbase::trace << "The information symbols are located in columns:\n";
   for (int loop = 0; loop < this->dim_k; loop++)
      {
      libbase::trace << this->info_symb_pos(loop) + 1 << " ";
      }
   libbase::trace << "\n";
   return sout.str();
   }

// object serialization - writing

/*
 *  This method write out the following format
 *
 * ldpc<gf<m,n>>
 * version
 * max_iter
 * n m
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col
 * vals of each non-zero entry per col
 *
 * where
 * - ldpc<gf<n,m>> is actually written by the serialisation code and not this method
 * - version is the file format version used
 * - max_iter is the maximum number of iterations used by the decoder
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway.
 *
 * given the matrix (5,3) over GF(4):
 *
 * 2 1 0 0 1
 * 0 2 2 3 3
 * 3 0 2 1 0
 *
 * The output would be
 * ldpc<gf<2,0x7>>
 * 1
 * 10
 * 5 3
 * 2 2
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
template <class GF_q> std::ostream& ldpc<GF_q>::serialize(std::ostream& sout) const
   {

   assertalways(sout.good());
   sout << "#version of this fileformat\n";
   sout << 1 << "\n";
   sout << "# number of iterations\n";
   sout << this->max_iter << "\n";
   sout << "# length n and dimension m\n";
   sout << this->length_n << " " << this->dim_pchk << "\n";
   sout << "#max col weight and max row weight\n";
   sout << this->max_col_weight << " " << this->max_row_weight << "\n";
   sout << "#the column weight vector\n";
   sout << this->col_weight << "\n";
   sout << "#the row weight vector\n";
   sout << this->row_weight << "\n";

   sout << "#the non zero pos per col\n";
   int num_of_non_zeros;
   int gf_val_int;
   int tmp_pos;
   int dim_gf = GF_q::dimension();
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      sout << this->M_n(loop1) << "\n";
      }
   // only output non-zero entries is the non-binary case
   if (dim_gf != 1)
      {
      libbase::vector<GF_q> non_zero_vals_in_col;
      sout << "#the non zero vals per col\n";
      for (int loop1 = 0; loop1 < this->length_n; loop1++)
         {
         num_of_non_zeros = this->M_n(loop1).size();
         non_zero_vals_in_col.init(num_of_non_zeros);
         for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
            {
            tmp_pos = this->M_n(loop1)(loop2) - 1;
            gf_val_int = this->pchk_matrix(tmp_pos, loop1);
            non_zero_vals_in_col(loop2) = gf_val_int;
            }
         sout << non_zero_vals_in_col << "\n";
         }
      }
   return sout;
   }

// object serialization - loading

/* loading of the serialized codec information
 * This method expects the following format
 *
 * ldpc<gf<m,n>>
 * version
 * max_iter
 * n m
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col
 * vals of each non-zero entry per col
 *
 * where
 * - version is the file format version used
 * - max_iter is the maximum number of iterations used by the decoder
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway.
 *
 *
 * given the matrix (5,3) over GF(4):
 *
 * 2 1 0 0 1
 * 0 2 2 3 3
 * 3 0 2 1 0
 *
 * An example file would be
 * ldpc<gf<2,0x7>>
 * 1
 * 10
 * 5 3
 * 2 2
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

template <class GF_q> std::istream& ldpc<GF_q>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   int version;
   sin >> libbase::eatcomments >> version;
   assertalways(version==1); //do something with this at some stage
   sin >> libbase::eatcomments >> this->max_iter;
   assertalways(this->max_iter>=1);

   sin >> libbase::eatcomments >> this->length_n;
   sin >> libbase::eatcomments >> this->dim_pchk;

   sin >> libbase::eatcomments >> this->max_col_weight;
   sin >> libbase::eatcomments >> this->max_row_weight;

   //read the col weights and ensure they are sensible
   this->col_weight.init(this->length_n);
   sin >> libbase::eatcomments;
   sin >> this->col_weight;

   assertalways((1<=this->col_weight.min())&&(this->col_weight.max()<=this->max_col_weight));

   //read the row weights and ensure they are sensible
   this->row_weight.init(this->dim_pchk);
   sin >> libbase::eatcomments;
   sin >> this->row_weight;
   assertalways((0<this->row_weight.min())&&(this->row_weight.max()<=this->max_row_weight));

   this->M_n.init(this->length_n);

   //read the non-zero entries pos per col
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      this->M_n(loop1).init(this->col_weight(loop1));
      sin >> libbase::eatcomments;
      sin >> this ->M_n(loop1);
      //ensure that the number of non-zero pos matches the previously read value
      assertalways(this->M_n(loop1).size()==this->col_weight(loop1));
      }

   //init the parity check matrix and read in the non-zero entries
   this->pchk_matrix.init(this->dim_pchk, this->length_n);
   this->pchk_matrix = 0.0;
   int tmp_entries;
   int tmp_pos;
   libbase::vector<GF_q> non_zero_vals;
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      tmp_entries = this->M_n(loop1).size();
      non_zero_vals.init(tmp_entries);
      if (GF_q::dimension() == 1)
         {
         //in the binary case the non-zero values are 1
         non_zero_vals = GF_q(1);
         }
      else
         {
         sin >> libbase::eatcomments;
         sin >> non_zero_vals;
         }
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         tmp_pos = this->M_n(loop1)(loop2) - 1;//we count from 0
         this->pchk_matrix(tmp_pos, loop1) = non_zero_vals(loop2);
         }
      }

   //derive the non-zero position per row from the parity check matrix
   this->N_m.init(this->dim_pchk);
   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      tmp_entries = this->row_weight(loop1);
      this->N_m(loop1).init(tmp_entries);
      tmp_pos = 0;
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
   this->init();
   return sin;
   }

/*
 * This method outputs the alist format of this code as
 * described by MacKay @
 * http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
 *
 * n m
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col followed by their values (for GF(p>2))
 * pos of each non-zero entry per row followed by their values (for GF(p>2))
 *
 * where
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - max_n is the maximum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 *
 * Note that the last set of positions is only used to
 * verify the information provided by the first set.
 * Also note that the alist format expects cols/rows with weight less than
 * the max col/row weight to be padded with extra 0s, eg
 * if a code has max col weight of 5 and a given col only has weight 3 then
 * it would look like
 * 1 4 5 0 0 (followed by the non-zero values in case of non-binary code)
 * These additional 0 need to be ignored
 *
 */

template <class GF_q> std::ostream& ldpc<GF_q>::write_alist(std::ostream& sout) const
   {

   assertalways(sout.good());

   // alist format version
   sout << this->length_n << " " << this->dim_pchk << "\n";
   sout << this->max_col_weight << " " << this->max_row_weight << "\n";
   this->col_weight.serialize(sout, ' ');
   this->row_weight.serialize(sout, ' ');
   int num_of_non_zeros;
   int gf_val_int;
   int tmp_pos;
   int dim_gf = GF_q::dimension();

   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      num_of_non_zeros = this->M_n(loop1).size().length();
      for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
         {
         sout << this->M_n(loop1)(loop2) << " ";
         }
      //add 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_col_weight - num_of_non_zeros); loop2++)
         {
         sout << "0 ";
         }

      // only output non-zero entries is the non-binary case
      if (dim_gf != 1)
         {
         for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
            {
            tmp_pos = this->M_n(loop1)(loop2) - 1;
            gf_val_int = this->pchk_matrix(tmp_pos, loop1);
            sout << gf_val_int << " ";
            }
         }
      sout << "\n";
      }

   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      num_of_non_zeros = this->N_m(loop1).size().length();
      for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
         {
         sout << this->N_m(loop1)(loop2) << " ";
         }
      //add 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_row_weight - num_of_non_zeros); loop2++)
         {
         sout << "0 ";
         }

      // only output non-zero entries is the non-binary case
      if (dim_gf != 1)
         {
         for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++)
            {
            tmp_pos = this->N_m(loop1)(loop2) - 1;
            gf_val_int = this->pchk_matrix(loop1, tmp_pos);
            sout << gf_val_int << " ";
            }
         }
      sout << "\n";
      }

   return sout;
   }

/* loading of the  alist format of an LDPC code
 * This method expects the following format
 *
 * n m
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col followed by their values
 * pos of each non-zero entry per row followed by their values
 *
 * where
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 * Note that the last set of positions and values are only used to
 * verify the information provided by the first set.
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway. An example row with 4 non-zero entries would look like this (assuming gf<3,0xB>
 * 1 3 9 10 1 3 7 2
 * ie the non-zero entries at pos 1,3,9 and 10 are 1,3,7 and 2 respectively
 *
 * Similarly a column with 3 non-zero entries over gf<3,0xB> would look like:
 * 3 6 12 4 3 6
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
 * 1 4 5 0 0 (followed by the non-zero values in case of non-binary code)
 * These additional 0 need to be ignored
 *
 */

template <class GF_q> std::istream& ldpc<GF_q>::read_alist(std::istream& sin)
   {

   assertalways(sin.good());

   sin >> libbase::eatcomments >> this->length_n;
   sin >> libbase::eatcomments >> this->dim_pchk;

   sin >> libbase::eatcomments >> this->max_col_weight;
   sin >> libbase::eatcomments >> this->max_row_weight;

   //read the col weights and ensure they are sensible
   int tmp_col_weight;
   this->col_weight.init(this->length_n);
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      sin >> libbase::eatcomments >> tmp_col_weight;
      //is it between 1 and max_col_weight?
      assertalways((1<=tmp_col_weight)&&(tmp_col_weight<=this->max_col_weight));
      this ->col_weight(loop1) = tmp_col_weight;
      }

   //read the row weights and ensure they are sensible
   int tmp_row_weight;
   this->row_weight.init(this->dim_pchk);
   for (int loop1 = 0; loop1 < this->dim_pchk; loop1++)
      {
      sin >> libbase::eatcomments >> tmp_row_weight;
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
   GF_q tmp_val = GF_q(1); //this is the default value for the binary case
   int gf_val_int;
   int dim_gf = GF_q::dimension();
   for (int loop1 = 0; loop1 < this->length_n; loop1++)
      {
      //read in the non-zero row entries
      tmp_entries = this->col_weight(loop1);
      this->M_n(loop1).init(tmp_entries);
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos;
         this->M_n(loop1)(loop2) = tmp_pos;
         tmp_pos--;//we start counting at 0 internally
         assertalways((0<=tmp_pos)&&(tmp_pos<this->dim_pchk));
         }
      //discard any padded 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_col_weight - tmp_entries); loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos;
         assertalways(tmp_pos==0);
         }
      //read in the non-zero values at those positions
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         tmp_pos = this->M_n(loop1)(loop2) - 1;
         if (dim_gf != 1)
            {
            //in the non-binary case we need to actually need to know the values
            //of the non-zero entries - in the binary case they are always 1
            sin >> libbase::eatcomments >> gf_val_int;
            tmp_val = GF_q(gf_val_int);
            assertalways ((GF_q(0))!=tmp_val);//ensure the value is non-zero
            }
         this->pchk_matrix(tmp_pos, loop1) = tmp_val;
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
         sin >> libbase::eatcomments >> tmp_pos;
         this->N_m(loop1)(loop2) = tmp_pos;
         tmp_pos--;//we start counting at 0 internally
         assertalways((0<=tmp_pos)&&(tmp_pos<this->length_n));
         }
      //discard any padded 0 zeros if necessary
      for (int loop2 = 0; loop2 < (this->max_row_weight - tmp_entries); loop2++)
         {
         sin >> libbase::eatcomments >> tmp_pos;
         assertalways(tmp_pos==0);
         }

      //verify that the non-zero values correspond to the
      //non-zero entries in the matrix
      for (int loop2 = 0; loop2 < tmp_entries; loop2++)
         {
         tmp_pos = this->N_m(loop1)(loop2) - 1;
         if (dim_gf != 1)
            {
            //in the non-binary case we need to actually need to know the values
            //of the non-zero entries - in the binary case they are always 1
            sin >> libbase::eatcomments >> gf_val_int;
            tmp_val = GF_q(gf_val_int);
            }
         assertalways(tmp_val==this->pchk_matrix(loop1,tmp_pos));
         }
      }

   this->init();
   return sin;
   }

}//end namespace

//Explicit realisations
namespace libcomm {
using libbase::serializer;

template class ldpc<gf<1, 0x3> > ;
template <>
const serializer ldpc<gf<1, 0x3> >::shelper = serializer("codec",
      "ldpc<gf<1,0x3>>", ldpc<gf<1, 0x3> >::create);

template class ldpc<gf<3, 0xB> > ;
template <>
const serializer ldpc<gf<3, 0xB> >::shelper = serializer("codec",
      "ldpc<gf<3,0xB>>", ldpc<gf<3, 0xB> >::create);

template class ldpc<gf<4, 0x13> > ;
template <>
const serializer ldpc<gf<4, 0x13> >::shelper = serializer("codec",
      "ldpc<gf<4,0x13>>", ldpc<gf<4, 0x13> >::create);

}
