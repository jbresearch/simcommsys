/*!
 * \file
 * $Id: tvb.cpp 9909 2013-09-23 08:43:23Z jabriffa $
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#include "conv.h"
#include "algorithm/fba2-factory.h"
#include "sparse.h"
#include "timer.h"
#include "cputimer.h"
#include "pacifier.h"
#include "vectorutils.h"
#include <sstream>
#include <iostream>

#include <bitset>
#include <limits>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show codebooks on frame advance
// 3 - Show transmitted sequence and the encoded message it represents
// 4 - Show prior and posterior sof/eof probabilities when decoding
// 5 - Show prior and posterior symbol probabilities when decoding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class sig, class real, class real2>
void conv<sig, real, real2>::advance() const
   {
   }

// encoding and decoding functions

template <class sig, class real, class real2>
void conv<sig, real, real2>::domodulate(const int N, const array1i_t& encoded,
      array1s_t& tx)
   {
   //Display encoded stream
   for(int z = 0; z < encoded.size();z++)
      {
      std::cout << encoded(z) << std::endl;
      }
   
   //TODO: Check Setting transmission size, no. of encoded bits *
   int tx_size = (ceil((double)(encoded.size()/k)))*n;
   tx.init(tx_size);

   int tx_cnt = 0;

   int out_loc = k+(no_states*2);
   int ns_loc = k+no_states;//next state location

   std::string curr_state = "";

   std::string input_and_state = "";

   for(int s = 0; s < no_states;s++)
      {
      curr_state = curr_state + "0";
      }
   
   //for(int i = 0; i < encoded.size(); i++)
   std::string input = "";
   int i = 0;
   int ab = encoded.size();
   while(i < encoded.size())
      {
      input = "";
      for(int j = 0;j < k;j++)
         {
         if(i > encoded.size())
            input += "0";
         else
            input += toString(encoded(i));

         i++;
         }

      input_and_state = input + curr_state;
      int row = bin2int(input_and_state);
      for(int out_cnt = 0; out_cnt < n; out_cnt++)
         {
         tx(tx_cnt) = statetable(row, out_loc+out_cnt);
         tx_cnt++;
         }
      
      for(int cnt = 0; cnt < no_states; cnt++)
         {
         curr_state[cnt] = toChar(statetable(row,ns_loc+cnt));
         }
      }
      
      std::cout << "Encoded" << std::endl;
      for(int d = 0; d < tx.size();d++)
         {
         std::cout << tx(d) << " ";
         }  
   }

template <class sig, class real, class real2>
void conv<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, array1vd_t& ptable)
   {

   system("cls");
   /*BCJR Algorithm - BEGIN*/
   double recv[] = {0.3,0.1,-0.5,0.2,0.8,0.5,-0.5,0.3,0.1,-0.7,1.5,-0.4};
   
   libbase::matrix<std::vector<double>> gamma;
   gamma.init(pow(2,no_states), 6);

   for(int col = 0; col < gamma.size().cols(); col++)
      {
      for(int row = 0; row < pow(2,no_states); row++)
         {
         for(int i = 0; i < pow(2,no_states);i++)
            {
            gamma(row,col).push_back(-1.0);
            }
         }
      }

   int inp_combinations = pow(2,k);
   double Lc = 5.0;
   double temp_gamma = 0.0;
   int out_loc = 0;
   int recv_loc = 0;
   int state_table_row = 0;
   int next_state_loc = 0;
   std::string str_nxt_state = "";
   int _nxt_state = 0;
   /*Working Gamma - BEGIN*/
   for(int col = 0; col < gamma.size().cols(); col++)
      {
      for(int row = 0; row < pow(2,no_states); row++)
         {
         for(int input = 0; input < inp_combinations; input++)
            {
            temp_gamma = 0.0;
            out_loc = k+(2*no_states);
            recv_loc = col * n;

            state_table_row = bin2int(int2bin(input, k) + int2bin(row,no_states));

            str_nxt_state = "";
            next_state_loc = k + no_states;
            for(int cnt = 0; cnt < no_states;cnt++)
               {
               str_nxt_state += toChar(statetable(state_table_row,next_state_loc));
               next_state_loc++;
               }
            _nxt_state = bin2int(str_nxt_state);

            for(int output = 0; output < n;output++)
               {
               int a = toInt(statetable(state_table_row,out_loc));
               double b = recv[recv_loc];

               temp_gamma = temp_gamma + ( toInt(statetable(state_table_row,out_loc)) ) * (recv[recv_loc]);
               out_loc++;
               recv_loc++;
               }
            double test = exp((Lc/2)*(temp_gamma));
            gamma(_nxt_state,col)[row] = exp((Lc/2)*(temp_gamma));
            //gamma(_nxt_state,col).insert(gamma(_nxt_state,col).begin()+row,exp((Lc/2)*(temp_gamma)));
            }
         }
      }

   for(int col = 0; col < gamma.size().cols(); col++)
      {
      for(int row = 0; row < pow(2,no_states); row++)
         {
         std::cout << "Row: " << row << " " << "Col: " << col << std::endl;
         for(int i = 0; i < pow(2,no_states);i++)
            {
            std::cout << gamma(row,col)[i] << " ";
            }
         std::cout << std::endl;
         }
      }
   /*Working Gamma - END*/

   /*Working alphas - BEGIN*/
   libbase::matrix<double> alpha;
   alpha.init(pow(2,no_states),7);

   //Initialising alpha
   for(int row = 0; row < alpha.size().rows();row++)
      {
      for(int col = 0; col < alpha.size().cols();col++)
         {
         alpha(row,col) = 0.0;
         }
      }
   alpha(0,0) = 1.0;

   double alpha_total = 0.0;

   for(int col = 1; col < alpha.size().cols();col++)
      {
      alpha_total = 0.0;
      for(int row = 0; row < alpha.size().rows();row++)
         {
            for(int gamma_cnt = 0; gamma_cnt < pow(2,no_states);gamma_cnt++)
               {
               if(gamma(row,col-1)[gamma_cnt] != -1)
                  {
                  alpha(row,col) += gamma(row,col-1)[gamma_cnt] * alpha(gamma_cnt,col-1);
                  //alpha_total += alpha(row,col);
                  }
               }
            alpha_total += alpha(row,col);
         }
      for(int r = 0; r < alpha.size().rows();r++)//Normalisation
         {
         alpha(r,col) /= alpha_total;
         }
      }
   
   /**/
   std::cout << std::endl;
   std::cout << std::endl;

   for(int col = 0; col < alpha.size().cols();col++)
      {
      for(int row = 0; row < alpha.size().rows();row++)
         {
         std::cout << alpha(row,col) << " ";
         }
      std::cout << std::endl;
      }
   /**/
   /*Working alphas - END*/

   /*Working bets - BEGIN*/
   libbase::matrix<double> beta;
   beta.init(pow(2,no_states),7);

   //Initialising beta
   for(int row = 0; row < beta.size().rows();row++)
      {
      for(int col = 0; col < beta.size().cols();col++)
         {
         beta(row,col) = 0.0;
         }
      }
   beta(0,beta.size().cols()-1) = 1.0;

      std::cout << std::endl;
   std::cout << std::endl;

   for(int col = 0; col < beta.size().cols();col++)
      {
      for(int row = 0; row < beta.size().rows();row++)
         {
         std::cout << beta(row,col) << " ";
         }
      std::cout << std::endl;
      }

   inp_combinations = pow(2,k);
   state_table_row = 0;
   _nxt_state = 0;

   double beta_total = 0.0;

   for(int col = beta.size().cols()-1; col > 0; col--)
      {
      for(int row = 0; row > beta.size().rows();row++)
         {
         for(int input = 0; input < inp_combinations; input++)
            {
            state_table_row = bin2int(int2bin(input, k) + int2bin(row,no_states));

            str_nxt_state = "";
            next_state_loc = k + no_states;
            for(int cnt = 0; cnt < no_states;cnt++)
               {
               str_nxt_state += toChar(statetable(state_table_row,next_state_loc));
               next_state_loc++;
               }
            _nxt_state = bin2int(str_nxt_state);

            beta(row,col) += beta(_nxt_state,col+1)*gamma(_nxt_state,col)[row];
            }
         beta_total += beta(row,col); 
         }
      for(int r = 0; r < beta.size().rows();r++)//Normalisation
         {
         beta(r,col) /= beta_total;
         }
      }
   /*Working beta - END*/
   
   /**/
   std::cout << std::endl;
   std::cout << std::endl;

   for(int col = 0; col < beta.size().cols();col++)
      {
      for(int row = 0; row < beta.size().rows();row++)
         {
         std::cout << beta(row,col) << " ";
         }
      std::cout << std::endl;
      }
   /**/

   /*BCJR Algorithm - END*/

   //const array1vd_t app; // empty APP table
   //dodemodulate(chan, rx, app, ptable);
   }

template <class sig, class real, class real2>
void conv<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   // Initialize for known-start
   init(chan);
   // Shorthand for transmitted and received frame sizes
   const int tau = this->output_block_size();
   const int rho = rx.size();
   // Check that rx size is within valid range
   //assertalways(mtau_max >= abs(rho - tau));
   // Set up start-of-frame drift pdf (drift = 0)
   array1d_t sof_prior;
   sof_prior.init(mtau_max - mtau_min + 1);
   sof_prior = 0;
   sof_prior(0 - mtau_min) = 1;
   // Set up end-of-frame drift pdf (drift = rho-tau)
   array1d_t eof_prior;
   eof_prior.init(mtau_max - mtau_min + 1);
   eof_prior = 0;
   eof_prior(rho - tau - mtau_min) = 1;
   // Offset rx by mtau_max and pad to a total size of tau+mtau_max-mtau_min
   array1s_t r;
   r.init(tau + mtau_max - mtau_min);
   r.segment(mtau_max, rho) = rx;
   // Delegate
   array1d_t sof_post;
   array1d_t eof_post;
   demodulate_wrapper(chan, r, 0, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, libbase::size_type<libbase::vector>(-mtau_min));
   }

template <class sig, class real, class real2>
void conv<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const libbase::size_type<libbase::vector> lookahead,
      const array1d_t& sof_prior, const array1d_t& eof_prior,
      const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
      array1d_t& eof_post, const libbase::size_type<libbase::vector> offset)
   {
   // Initialize for given start distribution
   init(chan, sof_prior, offset);
   // TODO: validate priors have required size?
#ifndef NDEBUG
   std::cerr << "DEBUG (tvb): offset = " << offset << ", mtau_min = "
         << mtau_min << "." << std::endl;
#endif
   assert(offset == -mtau_min);
   // Delegate
   demodulate_wrapper(chan, rx, lookahead, sof_prior, eof_prior, app, ptable,
         sof_post, eof_post, offset);
   }

/*!
 * \brief Wrapper for calling demodulation algorithm
 *
 * This method assumes that the init() method has already been called with
 * the appropriate parameters.
 */
template <class sig, class real, class real2>
void conv<sig, real, real2>::demodulate_wrapper(const channel<sig>& chan,
      const array1s_t& rx, const int lookahead, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const int offset)
   {
   // Inherit block size from last modulation step
   const int N = this->input_block_size();
   // In cases with lookahead, extend app table if supplied
   libbase::cputimer t1("t_priors");
   array1vd_t app_x;
   if (lookahead > 0 && app.size() > 0)
      {
      // Initialise extended app table (one symbol per timestep)
      assert(lookahead % n == 0);
      libbase::allocate(app_x, N + lookahead / n, q);
      app_x = 1.0; // equiprobable
      // Copy supplied prior to initial segment
      assert(app.size() == N);
      app_x.segment(0, N) = app;
      }
   else
      app_x = app;
   this->add_timer(t1);
   // Initialize FBA metric computer as needed
   if (changed_encoding_table)
      {
      libbase::cputimer te("t_enctable");
      fba_ptr->get_receiver().init(encoding_table);
      changed_encoding_table = false;
      this->add_timer(te);
      }
   // Call FBA and normalize results
#if DEBUG>=4
   using libbase::index_of_max;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "max at " << index_of_max(sof_prior) - offset << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
   std::cerr << "max at " << index_of_max(eof_prior) - offset << std::endl;
#endif
#if DEBUG>=5
   std::cerr << "app = " << app << std::endl;
#endif
   array1vr_t ptable_r;
   array1r_t sof_post_r;
   array1r_t eof_post_r;
   fba_ptr->decode(*this, rx, sof_prior, eof_prior, app_x, ptable_r, sof_post_r,
         eof_post_r, offset);
   // In cases with lookahead, re-compute EOF posterior at actual frame boundary
   libbase::cputimer t2("t_posteriors");
   if (lookahead > 0)
      fba_ptr->get_drift_pdf(eof_post_r, N);
   libbase::normalize_results(ptable_r.extract(0, N), ptable);
   libbase::normalize(sof_post_r, sof_post);
   libbase::normalize(eof_post_r, eof_post);
   this->add_timer(t2);
#if DEBUG>=4
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "max at " << index_of_max(sof_post) - offset << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
   std::cerr << "max at " << index_of_max(eof_post) - offset << std::endl;
#endif
#if DEBUG>=5
   std::cerr << "ptable = " << ptable << std::endl;
#endif
   }

// Setup procedure

template <class sig, class real, class real2>
void conv<sig, real, real2>::init(const channel<sig>& chan,
      const array1d_t& sof_pdf, const int offset)
   {
//#ifndef NDEBUG
//   libbase::cputimer t("t_init");
//#endif
//   // Inherit block size from last modulation step (and include lookahead)
//   const int N = this->input_block_size() + lookahead;
//   const int tau = N * n;
//   assert(N > 0);
//   // Copy channel for access within R()
//   mychan = dynamic_cast<const qids<sig, real2>&> (chan);
//   // Set channel block size to q-ary symbol size
//   mychan.set_blocksize(n);
//   // Set the probability of channel event outside chosen limits
//   mychan.set_pr(qids_utils::divide_error_probability(Pr, N));
//   // Determine required FBA parameter values
//   // No need to recompute mtau_min/max if we are given a prior PDF
//   mtau_min = -offset;
//   mtau_max = sof_pdf.size() - offset - 1;
//   if (sof_pdf.size() == 0)
//      mychan.compute_limits(tau, Pr, mtau_min, mtau_max, sof_pdf, offset);
//   int mn_min, mn_max;
//   mychan.compute_limits(n, qids_utils::divide_error_probability(Pr, N), mn_min,
//         mn_max);
//   int m1_min, m1_max;
//   mychan.compute_limits(1, qids_utils::divide_error_probability(Pr, tau),
//         m1_min, m1_max);
//   checkforchanges(m1_min, m1_max, mn_min, mn_max, mtau_min, mtau_max);
//   //! Determine whether to use global storage
//   bool globalstore = false; // set to avoid compiler warning
//   const int required = fba_type::get_memory_required(N, n, q, mtau_min,
//         mtau_max, mn_min, mn_max);
//   switch (storage_type)
//      {
//      case storage_local:
//         globalstore = false;
//         break;
//
//      case storage_global:
//         globalstore = true;
//         break;
//
//      case storage_conditional:
//         globalstore = (required <= globalstore_limit);
//         checkforchanges(globalstore, required);
//         break;
//
//      default:
//         failwith("Unknown storage mode");
//         break;
//      }
//   // Create an embedded algorithm object of the correct type
//   const bool thresholding = th_inner > real(0) || th_outer > real(0);
//   fba_ptr = fba2_factory<recv_type, sig, real, real2>::get_instance(
//         thresholding, flags.lazy, globalstore);
//   // Initialize our embedded metric computer with unchanging elements
//   // (needs to happen before fba initialization)
//   fba_ptr->get_receiver().init(n, q, mychan);
//   // Initialize forward-backward algorithm
//   fba_ptr->init(N, n, q, mtau_min, mtau_max, mn_min, mn_max, m1_min, m1_max,
//         th_inner, th_outer);
//#ifndef NDEBUG
//   this->add_timer(t);
//#endif
   }

template <class sig, class real, class real2>
void conv<sig, real, real2>::init()
   {
   }

// Marker-specific setup functions

template <class sig, class real, class real2>
void conv<sig, real, real2>::set_thresholds(const real th_inner,
      const real th_outer)
   {
//   /*This::th_inner = th_inner;
//   This::th_outer = th_outer;
//   test_invariant();*/
   }

// description output

template <class sig, class real, class real2>
std::string conv<sig, real, real2>::description() const
   {
   std::ostringstream sout;
   //sout << "Time-Varying Block Code (" << n << "," << q << ", ";
   //switch (codebook_type)
   //   {
   //   case codebook_sparse:
   //      sout << "sparse codebook";
   //      break;

   //   case codebook_random:
   //      sout << "random codebooks";
   //      break;

   //   case codebook_user_sequential:
   //      sout << codebook_name << " codebook ["
   //            << codebook_tables.size().rows() << ", sequential]";
   //      break;

   //   case codebook_user_random:
   //      sout << codebook_name << " codebook ["
   //            << codebook_tables.size().rows() << ", random]";
   //      break;

   //   default:
   //      failwith("Unknown codebook type");
   //      break;
   //   }
   //switch (marker_type)
   //   {
   //   case marker_zero:
   //      sout << ", no marker";
   //      break;

   //   case marker_random:
   //      sout << ", random marker";
   //      break;

   //   case marker_user_sequential:
   //      sout << ", user [" << marker_vectors.size() << ", sequential]";
   //      break;

   //   case marker_user_random:
   //      sout << ", user [" << marker_vectors.size() << ", random]";
   //      break;

   //   default:
   //      failwith("Unknown marker sequence type");
   //      break;
   //   }
   //sout << ", thresholds " << th_inner << "/" << th_outer;
   //sout << ", Pr=" << Pr;
   //sout << ", normalized";
   //sout << ", batch interface";
   //if (flags.lazy)
   //   sout << ", lazy computation";
   //else
   //   sout << ", pre-computation";
   //switch (storage_type)
   //   {
   //   case storage_local:
   //      sout << ", local storage";
   //      break;

   //   case storage_global:
   //      sout << ", global storage";
   //      break;

   //   case storage_conditional:
   //      sout << ", global storage [≤" << globalstore_limit << " MiB]";
   //      break;

   //   default:
   //      failwith("Unknown storage mode");
   //      break;
   //   }
   //if (lookahead == 0)
   //   sout << ", no look-ahead";
   //else
   //   sout << ", look-ahead " << lookahead << " codewords";
   //sout << "), ";
   //if (fba_ptr)
   //   sout << fba_ptr->description();
   //else
   //   sout << "FBA object not initialized";
   return sout.str();
   }

// object serialization - saving

template <class sig, class real, class real2>
std::ostream& conv<sig, real, real2>::serialize(std::ostream& sout) const
   {
   //sout << "# Version" << std::endl;
   //sout << 10 << std::endl;
   //sout << "# Inner threshold" << std::endl;
   //sout << th_inner << std::endl;
   //sout << "# Outer threshold" << std::endl;
   //sout << th_outer << std::endl;
   //sout << "# Probability of channel event outside chosen limits" << std::endl;
   //sout << Pr << std::endl;
   //sout << "# Lazy computation of gamma?" << std::endl;
   //sout << flags.lazy << std::endl;
   //sout << "# Storage mode for gamma (0=local, 1=global, 2=conditional)" << std::endl;
   //sout << storage_type << std::endl;
   //if (storage_type == storage_conditional)
   //   {
   //   sout << "#: Memory threshold for global storage (in MiB)" << std::endl;
   //   sout << globalstore_limit << std::endl;
   //   }
   //sout << "# Number of codewords to look ahead when stream decoding"
   //      << std::endl;
   //sout << lookahead << std::endl;
   //sout << "# n" << std::endl;
   //sout << n << std::endl;
   //sout << "# q" << std::endl;
   //sout << q << std::endl;
   //sout << "# codebook type (0=sparse, 1=random, 2=user[seq], 3=user[ran])"
   //      << std::endl;
   //sout << codebook_type << std::endl;
   //switch (codebook_type)
   //   {
   //   case codebook_sparse:
   //   case codebook_random:
   //      break;

   //   case codebook_user_sequential:
   //   case codebook_user_random:
   //      sout << "#: codebook name" << std::endl;
   //      sout << codebook_name << std::endl;
   //      sout << "#: codebook count" << std::endl;
   //      sout << num_codebooks() << std::endl;
   //      assert(num_codebooks() >= 1);
   //      assert(codebook_tables.size().cols() == q);
   //      for (int i = 0; i < num_codebooks(); i++)
   //         {
   //         sout << "#: codebook entries (table " << i << ")" << std::endl;
   //         for (int d = 0; d < q; d++)
   //            {
   //            codebook_tables(i, d).serialize(sout, ' ');
   //            //sout << std::endl;
   //            }
   //         }
   //      break;

   //   default:
   //      failwith("Unknown codebook type");
   //      break;
   //   }
   //sout << "# marker type (0=zero, 1=random, 2=user[seq], 3=user[ran])"
   //      << std::endl;
   //sout << marker_type << std::endl;
   //switch (marker_type)
   //   {
   //   case marker_zero:
   //   case marker_random:
   //      break;

   //   case marker_user_sequential:
   //   case marker_user_random:
   //      sout << "#: marker vectors" << std::endl;
   //      sout << marker_vectors.size() << std::endl;
   //      for (int i = 0; i < marker_vectors.size(); i++)
   //         {
   //         marker_vectors(i).serialize(sout, ' ');
   //         //sout << std::endl;
   //         }
   //      break;

   //   default:
   //      failwith("Unknown marker sequence type");
   //      break;
   //   }
   return sout;
   }

template <class sig, class real, class real2>
std::istream& conv<sig, real, real2>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> type >> libbase::verify;

   sin >> libbase::eatcomments >> k >> libbase::verify;

   sin >> libbase::eatcomments >> n >> libbase::verify;
   
   sin >> libbase::eatcomments;

   //FeedForward
   if(type == 0)
      {
      ffcodebook.init(k,n);
      std::string temp;
      
      std::string* ff_arr = new std::string[n*k];
      
      int * m_arr = new int[k];
      //Initializing m_arr to all 0
      for(int i = 0; i < k; i++)
         m_arr[i] = 0;

      //Getting m for each input, and setting m
      int str_size = 0;
      int cnt = 0;
      for(int row = 0; row < k; row++)
         {
         for(int col = 0; col < n; col++)
            {
            sin >> temp;
            ff_arr[cnt] = temp;
            cnt++;
            str_size = oct2bin(temp,0,0).size()-1;
            if(str_size > m_arr[row])
               m_arr[row] = str_size;
            }
         }

      cnt = 0;
      for(int row = 0; row < k; row++)
         {
         for(int col = 0; col < n; col++)
            {
            std::cout << ff_arr[cnt] << std::endl;
            std::cout << oct2bin(ff_arr[cnt], m_arr[row]+1, type) << std::endl;
            ffcodebook(row,col) = oct2bin(ff_arr[cnt], m_arr[row]+1, type);
            cnt++;
            std::cout << ffcodebook(row,col) << std::endl;
            }
         }

      fill_state_diagram_ff(m_arr);

      }

   //Feedback
   if(type > 0)
      {
      ffcodebook.init(k,n);
      fbcodebook.init(k,n);  

      std::string temp;
      m = 0;
      std::string* ff_arr = new std::string[n*k];
      std::string* fb_arr = new std::string[n*k];

      for(int cnt = 0; cnt < (k*n); cnt++)
         {
            sin >> temp;
            ff_arr[cnt] = temp;
            if(oct2bin(temp,0,0).size() > m)
               m = oct2bin(temp,0,0).size();
         }

      sin >> libbase::eatcomments;

      for(int cnt = 0; cnt < (k*n); cnt++)
         {
            sin >> temp;
            fb_arr[cnt] = temp;
            if(oct2bin(temp,0,0).size() > m)
               m = oct2bin(temp,0,0).size();
         }

      int counter = 0;
      for(int row = 0; row < k; row++)
         {
         for(int col = 0; col < n; col++)
            {            
            ffcodebook(row,col) = oct2bin(ff_arr[counter],m, type);
            counter++;
            std::cout << ffcodebook(row,col) << std::endl;
            }
         }

      counter = 0;
      for(int row = 0; row < k; row++)
         {
         for(int col = 0; col < n; col++)
            {
            fbcodebook(row,col) = oct2bin(fb_arr[counter],m, type);
            counter++;
            std::cout << fbcodebook(row,col) << std::endl;
            }
         }
      delete[] fb_arr;
      delete[] ff_arr;
      m--;
      fill_state_diagram_fb();
      }

   //init();
   return sin;
   }

template <class sig, class real, class real2>
void conv<sig,real,real2>::fill_state_diagram_ff(int *m_arr)
   {
   //Getting the number of states
   no_states = 0;

   for(int i=0;i<k;i++)
      no_states+=m_arr[i];
   
   //Fill the state table
   int comb = k+no_states;
   int no_rows = pow(2,comb);
   int no_cols = k+(2*no_states)+n;

   statetable.init(no_rows,no_cols);

   bool currbit = 0;
   int cnt_rows = no_rows;
   int counter = cnt_rows/2;
   //Filling input and current state
   for(int col = 0; col < no_cols; col++)
      {
      cnt_rows /= 2;
      counter = cnt_rows;
      currbit = 0;
      for(int row = 0; row < no_rows; row++)
         {
         if(col >= comb)
            statetable(row,col)=currbit;
         else
            {
            if(counter == 0)
               {
               counter = cnt_rows;
               currbit = !currbit;
               }
            statetable(row,col)=currbit;
            counter--;
            }
         }
      }

   //Fill output and next state
   int state_begin = k;
   int state_end = k+m_arr[0]-1;
   int state_cnt = state_begin;
   bool temp_out = 0;
   int out_start = k+(no_states*2);
   for(int inp_cnt = 0; inp_cnt < k; inp_cnt++)
      {
         for(int row = 0; row < no_rows; row++)
            {
            //Working outputs
            for(int out_cnt = 0; out_cnt < n; out_cnt++)
               {
               temp_out = 0;
               state_cnt = state_begin;
               //Working effect of input
               bool a = statetable(row,inp_cnt);
               bool b = toBool(ffcodebook(inp_cnt,out_cnt)[0]);
               temp_out = temp_out ^ (statetable(row,inp_cnt) & toBool(ffcodebook(inp_cnt,out_cnt)[0]));
               //Working effect of shift registers
               for(int conn_cnt = 1; conn_cnt < (m_arr[inp_cnt]+1); conn_cnt++)
                  {
                     bool c = statetable(row,state_cnt);
                     bool d = toBool(ffcodebook(inp_cnt,out_cnt)[conn_cnt]);
                     temp_out = temp_out ^ (statetable(row,state_cnt) & toBool(ffcodebook(inp_cnt,out_cnt)[conn_cnt]));
                     state_cnt++;
                  }
               statetable(row,out_cnt+out_start) ^= temp_out;
               }
            //Working next state
            state_cnt = state_begin;
            statetable(row,state_cnt+no_states) = statetable(row,inp_cnt);
            //state_begin++;
            for(int i = 0; i < m_arr[inp_cnt]-1;i++)
               {
               statetable(row, state_cnt+no_states+1) = statetable(row,state_cnt);
               state_cnt++;
               }
            }
         state_begin = state_end + 1;
         state_end = state_end + m_arr[inp_cnt]-1;
      }

   disp_statetable();
   }
   
template <class sig, class real, class real2>
void conv<sig,real,real2>::fill_state_diagram_fb()
   {
   //Fill the state table
   no_states = m;
   int comb = k+m;
   int no_rows = pow(2,comb);
   int no_cols = k+(2*m)+n;

   statetable.init(no_rows,no_cols);

   bool currbit = 0;
   int cnt_rows = no_rows;
   int counter = cnt_rows/2;
   //Filling input and current state
   for(int col = 0; col < no_cols; col++)
      {
      cnt_rows /= 2;
      counter = cnt_rows;
      currbit = 0;
      for(int row = 0; row < no_rows; row++)
         {
         if(col >= comb)
            statetable(row,col)=currbit;
         else
            {
            if(counter == 0)
               {
               counter = cnt_rows;
               currbit = !currbit;
               }
            statetable(row,col)=currbit;
            counter--;
            }
         }
      }
   

   //Filling Output and Next State
   if(type == 1)
      {
      bool feedback = 0;
      bool feedback_worked = 0;
      int output_col = k+2*m;//location of first output index
      int input_col = 0;
      for(int row = 0; row < no_rows; row++)
         {
         feedback = 0;
         feedback_worked = 0;
         output_col = k+2*m;//location of first output index
         input_col = 0;
         for(int out_cnt = 0; out_cnt < n; out_cnt++)
            {
            //Working outputs
            //check if it has feedback
            if(bin2int(fbcodebook(input_col,out_cnt)) == 0)//no feedback
               {
               statetable(row,output_col) = statetable(row,input_col);
               }
            else//has feedback
               {
               //Working the feedback
               if(feedback_worked == 0)
                  {
                  for(int i = 0; i < (m+1);i++)
                     {
                     bool x = toBool(fbcodebook(input_col,out_cnt)[i]);
                     bool y = statetable(row,i);
                     feedback = feedback ^ (toBool(fbcodebook(input_col,out_cnt)[i]) & statetable(row,i));
                     }
                  feedback_worked = 1;
                  }

               //Working the output
               bool output = 0;
               for(int i = 0; i < (m+1);i++)
                  {
                  if(i == 0)//Replace input with feedback result
                     {
                     bool x = toBool(ffcodebook(input_col,out_cnt)[i]);
                     bool y = feedback;
                     output = output ^ (toBool(ffcodebook(input_col,out_cnt)[i]) & feedback);
                     }
                  else
                     {
                     bool x = toBool(ffcodebook(input_col,out_cnt)[i]);
                     bool y = statetable(row,(k+i)-1);
                     output = output ^ (toBool(ffcodebook(input_col,out_cnt)[i]) & statetable(row,(k+i)-1));
                     }
                  }
               statetable(row,output_col) = output;
               }
            output_col++;
            }
         //Working next state
         for(int i = 0; i < m;i++)
            {
            statetable(row,(i+m+1)) = statetable(row,i); 
            }
         if(feedback_worked == 1)
            statetable(row,(m+1)) = feedback;
         }
      }
   else if (type == 2)
      {
      bool feedback = 0;
      int output_col = k+2*m;//location of first output index
      int input_col = 0;
      bool output = 0;
      bool temp_out = 0;

      int fb_loc = 0;

      for(int row = 0; row < no_rows; row++)
         {
         /*Resetting every iteration*/
         feedback = 0;
         output_col = k+2*m;//location of first output index
         input_col = 0;
         //output = 0;
         temp_out = 0;
         
         //Working the outputs
         for(int out_cnt =0;out_cnt < n; out_cnt++)
            {
            for(int inp_cnt = 0; inp_cnt < k; inp_cnt++)
               {
               int state_loc = (k+m)-1; 
               temp_out = 0;
               if(bin2int(fbcodebook(inp_cnt,out_cnt)) == 0)//no feedback
                  {
                  for(int i = 0; i < (m+1); i++)
                     {
                     if(i < m) //Getting the states
                        {
                        bool x = toBool(ffcodebook(inp_cnt,out_cnt)[i]);
                        bool y = statetable(row,state_loc);
                        temp_out = temp_out ^ (toBool(ffcodebook(inp_cnt,out_cnt)[i]) & statetable(row,state_loc));
                        state_loc--;   
                        }
                     else //Getting the relevant input
                        {
                        bool x = toBool(ffcodebook(inp_cnt,out_cnt)[i]);
                        bool y = statetable(row,inp_cnt);
                        temp_out = temp_out ^ (toBool(ffcodebook(inp_cnt,out_cnt)[i]) & statetable(row,inp_cnt));
                        }
                     }
                  output = output ^ temp_out;
                  }
               else//has feedback
                  {
                  fb_loc = out_cnt;
                  bool x = toBool(ffcodebook(inp_cnt,out_cnt)[m]);
                  bool y = statetable(row,inp_cnt);
                  output = output ^ (toBool(ffcodebook(inp_cnt,out_cnt)[m]) & statetable(row,inp_cnt));
                  if(inp_cnt == (k-1))
                     {
                     output = output ^ statetable(row,(k+m)-1);
                     feedback = output;
                     }
                  }
               }
            statetable(row,output_col) = output;
            output = 0;
            output_col++;
            }

            //Working next state
            int state_loc = k+m;
            bool state = 0;
            int inp_cnt = 0;
            for(int i = 0; i < m; i++)
               {
               for(inp_cnt = 0; inp_cnt < k; inp_cnt++)
                  {
                  bool x = toBool(ffcodebook(inp_cnt,fb_loc)[i]);
                  bool y = statetable(row,inp_cnt);
                  state = state ^ ( toBool(ffcodebook(inp_cnt,fb_loc)[i]) & statetable(row,inp_cnt));
                  }
               bool t = toBool(fbcodebook(0,fb_loc)[i]);
               state = state ^ ( toBool(fbcodebook(0,fb_loc)[i]) & feedback);
               //check whether it has previous state
               if((state_loc-1) >= (k+m))
                  {
                  bool a = statetable(row, (state_loc-m-1));
                  state = state ^ statetable(row, (state_loc-m-1));
                  }
               statetable(row,state_loc) = state;
               state = 0;
               state_loc++;
               }
         }
      }

   disp_statetable();

   }

/*This function converts an integer to a binary String stream.
Input is the integer that needs to be converted
Size is the number of bits that you want the result
Ex. Converting the integer 1, with size 3 will give 001 instead of just 1*/
template <class sig, class real, class real2>
std::string conv<sig, real, real2>::oct2bin(std::string input, int size, int type)
   {
   int div = 0;
   int rem = 0;
   
   //From octal to decimal
   int counter = 0;
   for(int i = input.length()-1; i >= 0; i--)
      {
      div = div + (((int)input[i]-48)*pow(8.0,counter));
      counter++;
      }
   //From decimal to string
   std::string binary_stream = "";
   std::stringstream out;
	
   while(1)
      {
      rem = div % 2;
      div = (int) floor( static_cast<double>(div / 2));

      out << rem;

      binary_stream += out.str();
      out.str("");

      if(div == 0)
         {
         reverse(binary_stream.begin(), binary_stream.end());
	 
         if(binary_stream.size() != size && size > 0)
            {
            int bitstream_size = binary_stream.size();
				
            for(int i = 0; i < size - bitstream_size; i++)
               {
               if(type == 1)
                  binary_stream = binary_stream + "0";
               else
                  binary_stream = "0" + binary_stream;
               }
            }

         return binary_stream;
         }
      }
   }

template <class sig, class real, class real2>
int conv<sig, real, real2>::bin2int(std::string binary)
{
	int result = 0;

	reverse(binary.begin(), binary.end());

	for(unsigned int i = 0; i < binary.size();i++)
	{
		if(binary[i] == '1')
		{
			result = result + (int) pow(2,i);
		}
	}
	return result;
}

template <class sig, class real, class real2>
std::string conv<sig, real, real2>::int2bin(int input, int size)
   {
   std::string binary_stream = "";
   std::stringstream out;

   int div = input;
   int rem = 0;
	
   while(1)
      {
      rem = div % 2;
      div = (int) floor((double)(div / 2));

      out << rem;

      binary_stream += out.str();
      out.str("");

      if(div == 0)
         {
         reverse(binary_stream.begin(), binary_stream.end());
			
         if(binary_stream.size() != size)
            {
            int bitstream_size = binary_stream.size();
				
            for(int i = 0; i < size - bitstream_size; i++)
               {
               binary_stream = "0" + binary_stream;
               }
            }
         return binary_stream;
         }
      }
   }

template <class sig, class real, class real2>
bool conv<sig, real, real2>::toBool(char const& bit)
   {
     return bit != '0';
   }

template<class sig, class real, class real2>
void conv<sig,real,real2>::disp_statetable()
   {
   system("cls");
   for(int row = 0; row < statetable.size().rows(); row++)
      {
      for(int col = 0; col < statetable.size().cols(); col++)
         {
         std::cout << statetable(row,col) << " ";
         }
      std::cout << std::endl;
      }
   }

template<class sig, class real, class real2>
std::string conv<sig,real,real2>::toString(int number)
   {
   std::stringstream ss;//create a stringstream
   ss << number;
   return ss.str();
   }

template<class sig, class real, class real2>
char conv<sig,real,real2>::toChar(bool bit)
   {
   if(bit)
      return '1';
   else
      return '0';
   }

template<class sig, class real, class real2>
int conv<sig,real,real2>::toInt(bool bit)
   {
   if(bit)
      return 1;
   else
      return -1;
   }
} // end namespace

#include "gf.h"
#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define ALL_SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#ifdef USE_CUDA
#define REAL_PAIRS_SEQ \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#else
#define REAL_PAIRS_SEQ \
   ((mpgnu)(mpgnu)) \
   ((logrealfast)(logrealfast)) \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#endif

/* Serialization string: tvb<type,real,real2>
 * where:
 *      type = bool | gf2 | gf4 ...
 *      real = float | double | [logrealfast | mpgnu (CPU only)]
 *      real2 = float | double | [logrealfast | mpgnu (CPU only)]
 */
#define INSTANTIATE3(args) \
      template class conv<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer conv<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockmodem", \
            "conv<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            conv<BOOST_PP_SEQ_ENUM(args)>::create);

#define INSTANTIATE2(r, symbol, reals) \
      INSTANTIATE3( symbol reals )

#define INSTANTIATE1(r, symbol) \
      BOOST_PP_SEQ_FOR_EACH(INSTANTIATE2, symbol, REAL_PAIRS_SEQ)

// NOTE: we *have* to use for-each product here as we cannot nest for-each
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE1, (ALL_SYMBOL_TYPE_SEQ))

} // end namespace