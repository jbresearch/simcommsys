/*!
 * \file
 * $Id: tvb.cpp 10304 2013-11-28 14:14:36Z jabriffa $
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

#include "conv_modem.h"
#include "algorithm/fba2-factory.h"
#include "sparse.h"
#include "timer.h"
#include "cputimer.h"
#include "pacifier.h"
#include "vectorutils.h"
#include <sstream>
#include <vector>
/*For vector*/
#include <numeric>
#include <algorithm>
#include <functional>
/*For overflow*/
//#include <cfenv>
//#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>

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
void conv_modem<sig, real, real2>::advance() const
   {
   }

// encoding and decoding functions

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::domodulate(const int N, const array1i_t& encoded,
      array1s_t& tx)
   {
   //Checking that the block lenghts match
   //assert(encoded.size() == block_length);
   tx.init(block_length_w_tail);
   encode_data(encoded, tx);

   //std::cout << "Encoded" << std::endl;
   //for(int x = 0; x < encoded.size(); x++)
   //   std::cout << encoded(x) << " ";

   //std::cout << std::endl;
   //std::cout << "Transmitted" << std::endl;
   //for(int x = 0; x < tx.size(); x++)
   //   std::cout << tx(x) << " ";
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::encode_data(const array1i_t& encoded, array1s_t& tx)
   {

   //std::cout << std::endl;
   //std::cout << std::endl;
   //
   //std::cout << "Original Data" << std::endl;

   //for (int i = 0; i < encoded.size(); i++)
   //   {
   //   std::cout << encoded(i) << " ";
   //   }
   //std::cout << std::endl;
   


   int out_loc = k+(no_states*2);
   int ns_loc = k+no_states;//next state location
   int encoding_counter = 0;
   int tx_cnt = 0;
   int row = 0;
   
   std::string curr_state = "";
   std::string input_and_state = "";
   for(int s = 0; s < no_states;s++)
      curr_state = curr_state + "0";
   
   std::string input = "";
   
   while(encoding_counter < encoded.size())
      {
      input = "";
      //Getting input
      for(int inp_cnt = 0;inp_cnt < k;inp_cnt++)
         {
         if(encoding_counter > encoded.size())
            input += "0";
         else
            input += toString(encoded(encoding_counter));

         encoding_counter++;
         }

      input_and_state = input + curr_state;
      row = bin2int(input_and_state);
      //Encoding
      for(int out_cnt = 0; out_cnt < n; out_cnt++)
         {
         tx(tx_cnt) = statetable(row, out_loc+out_cnt);
         tx_cnt++;
         }
      
      //Changing current state
      for(int cnt = 0; cnt < no_states; cnt++)
         {
         curr_state[cnt] = toChar(statetable(row,ns_loc+cnt));
         }
      }

   //Tailing off
   for(int tailoff_cnt = 0; tailoff_cnt < m; tailoff_cnt++)
      {
      input = "";
      for(int inp_cnt = 0;inp_cnt < k;inp_cnt++)
         {
         input += "0";
         }

      input_and_state = input + curr_state;
      row = bin2int(input_and_state);
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
   
   //std::cout << "Encoded data" << std::endl;
   //print_sig(tx);
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::dodemodulate(const channel<sig>& chan, const array1s_t& rx, array1vd_t& ptable)
   {

   //std::feclearexcept(FE_OVERFLOW);
   //std::feclearexcept(FE_UNDERFLOW);


   //std::cout << "Overflow flag before: " << (bool)std::fetestexcept(FE_OVERFLOW) << std::endl;
   //std::cout << "Underflow flag before: " << (bool)std::fetestexcept(FE_UNDERFLOW) << std::endl;

   gamma_storage.clear();
   vector_3d().swap(gamma_storage);
   gamma_storage.resize(pow(2,no_states));

   //std::cout << std::endl;
   //std::cout << "Received" << std::endl;
   //for(int x = 0; x < rx.size(); x++)
   //   std::cout << rx(x) << " ";

   /*Timing - begin*/
   //long int before = GetTickCount();
   /*Timing - end*/

   mychan = dynamic_cast<const qids<sig, real2>&> (chan);
   mychan.set_blocksize(2);

   unsigned int no_insdels = no_del + no_ins + 1;

   unsigned int b_size = block_length + m + 1;

   unsigned int num_states = pow(2,no_states);

   vector<b_storage> b_vector(b_size, b_storage(num_states));

   /*Setting up the first alpha value - BEGIN*/
   b_vector[0].setmin_bs(0);
   //b_vector[0].state_bs_vector.resize(1);
   b_vector[0].state_bs_vector[0].push_back(state_bs_storage(1));
   /*Setting up the first alpha value - END*/

   unsigned int inp_combinations = pow(2,k);
   unsigned int next_state = 0;

   array1s_t orig_codeword;
   array1s_t recv_codeword;               

   double gamma = 0.0;
   double alpha = 0.0;
   double beta = 0.0;
   double alpha_total = 0.0;
   double beta_total = 0.0;
   unsigned int num_bs = 0;

   unsigned int cur_bs = 0;
   unsigned int next_bs = 0;

   unsigned int norm_b = 0;

   unsigned int drift = 0;
   unsigned int symb_shift = 0;//the current number of symbol shifts

   /*initialising outtable - BEGIN*/
   array1vd_t outtable;
   outtable.init(b_size-1);
   
   for(int cnt = 0; cnt < outtable.size(); cnt++)
      {
      outtable(cnt).init(2);
      outtable(cnt) = 0;
      }
   /*initialising outtable - END*/

   //For all decoded bits
   b_size--;
   for(unsigned int b = 0; b < b_size; b++)
      {
      //Resetting alpha and beta values every iteration
      alpha_total = 0.0;
      beta_total = 0.0;

      if(b >= (unsigned int) block_length) //this is tailing so input is always 0
         inp_combinations = 1;

      /*Taking care of tailing for last decoded bit*/
      if ((b + 1) == b_size)
         b_vector[b + 1].setmin_bs(b_size*n);
      else
         b_vector[b+1].setmin_bs(b_vector[b].getmin_bs() + n - no_del);
      
      //For all the number of states
      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[b].state_bs_vector[cur_state].size();
         
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            cur_bs = b_vector[b].getmin_bs() + cnt_bs;

            for(unsigned int input = 0; input < inp_combinations; input++)
               {
               next_state = get_next_state(input, cur_state);
               next_bs = cur_bs + n - no_del;//setting up the initial point of next_bs

               get_output(input, cur_state, orig_codeword);
               
               for(unsigned int cnt_next_bs = 0; cnt_next_bs < no_insdels; cnt_next_bs++)
                  {
                  /*Calculating the current drift - BEGIN*/
                  drift = abs(int (next_bs - (b+1)*n));
                  symb_shift = floor(double(drift)/double(n));
                  /*Calculating the current drift - END*/

                  //if(symb_shift <= rho)
                  if ( (((b + 1) == b_size) && drift == 0) || (((b+1) < b_size) && (symb_shift <= rho)))
                     {
                     get_received(b, cur_bs, next_bs, no_del, rx, recv_codeword);

                     //system("cls");
                     //std::cout << "Original" << std::endl;
                     //print_sig(orig_codeword);
                     //std::cout << "Received" << std::endl;
                     //print_sig(recv_codeword);

                     //Work gamma
                     /**/
                     //gamma = work_gamma(orig_codeword,recv_codeword);
                  
                     gamma = get_gamma(cur_state, cur_bs, next_state, next_bs, orig_codeword, recv_codeword);
                  
                  
                     //Work alpha
                     unsigned int st_cur_bs = (cur_bs-b_vector[b].getmin_bs());//the actual store location for current bs
                     unsigned int st_nxt_bs = (next_bs - b_vector[b+1].getmin_bs());//the actual store location for next bs
                     //For debugging
                     alpha = b_vector[b].state_bs_vector[cur_state][st_cur_bs].getalpha();
                     alpha = gamma * alpha;
                     alpha_total += alpha;
                     //For release
                     //alpha = gamma * b_vector[b].state_bs_vector[cur_state][cur_bs].getalpha();

                     //storing gamma
                     /*Check whether bit shift location is already available - Begin*/
                     if(b_vector[b+1].state_bs_vector[next_state].size() < (st_nxt_bs + 1))
                        b_vector[b+1].state_bs_vector[next_state].resize(st_nxt_bs + 1);
                     /*Check whether bit shift location is already available - End*/

                     b_vector[b+1].state_bs_vector[next_state][st_nxt_bs].gamma.push_back(Gamma_Storage(cur_state,st_cur_bs,gamma));
                     //storing alpha
                     b_vector[b+1].state_bs_vector[next_state][st_nxt_bs].setalpha(alpha);

                     next_bs++;//Incrementing next_bs
                     }
                  else
                     {
                     if (b_vector[b + 1].getmin_bs() == next_bs)
                        b_vector[b + 1].setmin_bs(++next_bs);
                     else
                        next_bs++;
                     }

                  }
               }
            }
         }

      //Normalisation
      norm_b = b+1;
      ////system("cls");
      //std::cout << "Before Norm" << std::endl;
      //std::cout << std::endl;

      //double total = 0.0;
      //for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
      //   {
      //   num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
      //   //For all the number of bitshifts available
      //   for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
      //      {
      //      std::cout << b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getalpha() << std::endl;
      //      total += b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getalpha();
      //      }
      //   }
      //std::cout << std::endl;
      //std::cout << "Total is: " << total << std::endl;

      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].normalpha(alpha_total);
            }
         }

      //total = 0.0;
      //std::cout << std::endl;
      //std::cout << "After Norm" << std::endl;
      //std::cout << std::endl;

      //for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
      //   {
      //   num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
      //   //For all the number of bitshifts available
      //   for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
      //      {
      //      std::cout << b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getalpha() << std::endl;
      //      total += b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getalpha();
      //      }
      //   }
      //std::cout << std::endl;
      //std::cout << "Total is: " << total << std::endl;

      }

   /*Setting up the first beta value - BEGIN*/
   /*int test = n * b_size;
   int test2 = b_vector[b_size].getmin_bs();

   int result_loc = test - test2;*/
   b_vector[b_size].state_bs_vector[0][(n*b_size) - b_vector[b_size].getmin_bs()].setbeta(1);
   /*Setting up the first beta value - END*/

   unsigned int size_gamma = 0;

   unsigned int prev_bs = 0;
   unsigned int prev_state = 0;
   
   //std::cout << "Beta and output" << std::endl;

   std::vector< std::vector<double> > vec_tmp_output;
   vec_tmp_output.resize(pow(2,k));

   double out_summation = 0.0;
   double temp_out = 0.0;

   for(unsigned int b = b_size; b > 0; b--)
      {
      //std::cout << b << std::endl;
      beta_total = 0.0;
      out_summation = 0.0;

      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[b].state_bs_vector[cur_state].size();
         
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            size_gamma = b_vector[b].state_bs_vector[cur_state][cnt_bs].gamma.size();

            for(unsigned int cnt_gamma = 0; cnt_gamma < size_gamma; cnt_gamma++)
               {
               //Getting previous state and bitshift
               prev_state = b_vector[b].state_bs_vector[cur_state][cnt_bs].gamma[cnt_gamma].getstate();
               prev_bs = b_vector[b].state_bs_vector[cur_state][cnt_bs].gamma[cnt_gamma].getbitshift();
               
               //Getting gamma
               gamma = b_vector[b].state_bs_vector[cur_state][cnt_bs].gamma[cnt_gamma].getgamma();
               
               //Getting alpha
               alpha = b_vector[b - 1].state_bs_vector[prev_state][prev_bs].getalpha();
               
               //Getting beta
               beta = b_vector[b].state_bs_vector[cur_state][cnt_bs].getbeta();
               
               //Working out the output
               //unsigned int inp = get_input(prev_state, cur_state);
               /*Inserting output values for normalisation - BEGIN*/
               temp_out = alpha * gamma * beta;
               vec_tmp_output[get_input(prev_state, cur_state)].push_back(temp_out);
               out_summation += temp_out;
               /*Inserting output values for normalisation - END*/
               //outtable(b - 1)(get_input(prev_state, cur_state)) += (alpha * gamma * beta);

               //Working out next beta
               beta = beta * gamma;
               beta_total += beta;
               b_vector[b-1].state_bs_vector[prev_state][prev_bs].setbeta(beta);
               }
            }
         }


      transform(vec_tmp_output[0].begin(), vec_tmp_output[0].end(), vec_tmp_output[0].begin(), bind2nd( divides<double>(), out_summation));
      transform(vec_tmp_output[1].begin(), vec_tmp_output[1].end(), vec_tmp_output[1].begin(), bind2nd( divides<double>(), out_summation));
      
      outtable(b-1)(0) = std::accumulate(vec_tmp_output[0].begin(), vec_tmp_output[0].end(), 0.0);
      outtable(b-1)(1) = std::accumulate(vec_tmp_output[1].begin(), vec_tmp_output[1].end(), 0.0);

      vec_tmp_output[0].clear();
      vec_tmp_output[1].clear();

      //Normalisation
      norm_b = b-1;
      ////system("cls");
      //std::cout << "Before Norm" << std::endl;
      //std::cout << std::endl;

      //double total = 0.0;
      //for(unsigned int cur_state  = 0; cur_state < num_states; cur_state++)
      //   {
      //   num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
      //   //For all the number of bitshifts available
      //   for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
      //      {
      //      std::cout << b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getbeta() << std::endl;
      //      total += b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getbeta();
      //      }
      //   }
      //std::cout << std::endl;
      //std::cout << "Total is: " << total << std::endl;

      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].normbeta(beta_total);
            }
         }

      

      //total = 0.0;
      //std::cout << std::endl;
      //std::cout << "After Norm" << std::endl;
      //std::cout << std::endl;

      //for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
      //   {
      //   num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
      //   //For all the number of bitshifts available
      //   for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
      //      {
      //      std::cout << b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getbeta() << std::endl;
      //      total += b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].getbeta();
      //      }
      //   }
      //std::cout << std::endl;
      //std::cout << "Total is: " << total << std::endl;

      }
   ptable = outtable.extract(0,block_length);
   //ptable = outtable;

   vec_tmp_output.clear();
   vector< vector<double> >().swap(vec_tmp_output);
   
   b_vector.clear();
   vector<b_storage>().swap(b_vector);

   //std::cout << std::endl;
   //std::cout << "Decoded" << std::endl;   
   //for(int k = 0; k < ptable.size(); k++)
   //   {
   //   if(ptable(k)(0) > ptable(k)(1))
   //      std::cout << "0 ";
   //   else
   //      std::cout << "1 ";
   //   }

   //std::cout << std::endl;
   //std::cout << std::endl;

   //for(int k = 0; k < ptable.size(); k++)
   //   std::cout << ptable(k) << std::endl;
      

   //std::cout << std::endl;
   //std::cout << std::endl;
   ///*Timing - begin*/
   //long int after = GetTickCount();
   //std::cout << "Time(ms): " << after-before << std::endl;
   //std::cout << "Time(s): " << (double) (after-before)/1000 << std::endl;
   //std::cout << "Time(min): " << (double) ((after-before)/1000)/60 << std::endl;
   /*Timing - end*/

   //if ((bool)std::fetestexcept(FE_OVERFLOW) == 1 || (bool)std::fetestexcept(FE_UNDERFLOW) == 1)
   //   {
   //   std::cout << "Overflow/underflow" << std::endl;
   //   std::cin.get();
   //   }

   //std::cout << "Overflow flag after: " << (bool)std::fetestexcept(FE_OVERFLOW) << std::endl;
   //std::cout << "Underflow flag after: " << (bool)std::fetestexcept(FE_UNDERFLOW) << std::endl;

   }



template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::print_sig(array1s_t& data)
   {
   for(int a = 0; a < data.size(); a++)
      std::cout << data(a) << " ";

   std::cout << std::endl;
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
  
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const libbase::size_type<libbase::vector> lookahead,
      const array1d_t& sof_prior, const array1d_t& eof_prior,
      const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
      array1d_t& eof_post, const libbase::size_type<libbase::vector> offset)
   {

   }

/*Current state, current bitshift, next state, next bs*/
template <class sig, class real, class real2>
double conv_modem<sig, real, real2>::get_gamma(unsigned int cur_state, unsigned int cur_bs, unsigned int next_state, unsigned int next_bs, array1s_t& orig_seq, array1s_t& recv_seq)
   {
  
   if(gamma_storage[cur_state].size() < (cur_bs + 1)) //checking if current bit-shift location exists, if not create one
      {
      //Does not exist
      gamma_storage[cur_state].resize(cur_bs + 1);
      }

   int search_cnt = gamma_storage[cur_state][cur_bs].size();

   for(int i = 0; i < search_cnt; i++)
      {
      if( (gamma_storage[cur_state][cur_bs][i].getstate() == next_state) && (gamma_storage[cur_state][cur_bs][i].getbitshift() == next_bs) )
         {
         return gamma_storage[cur_state][cur_bs][i].getgamma();//The value of gamma is found
         }
      }

   //The value of gamma is not found need to be worked out
   double gamma = work_gamma(orig_seq,recv_seq);
   gamma_storage[cur_state][cur_bs].push_back(Gamma_Storage(next_state, next_bs, gamma));

   return gamma;
   }

template <class sig, class real, class real2>
double conv_modem<sig, real, real2>::work_gamma(array1s_t& orig_seq, array1s_t& recv_seq)
   {
   
   //double P_err = mychan.get_ps();
   //double P_no_err = 1 - P_err;

   //int no_err = 0;

   //double gamma = 0.0;

   //if (orig_seq.size() == recv_seq.size())
   //   {
   //   //Calculating the Hamming distance
   //   for (int cnt = 0; cnt < orig_seq.size(); cnt++)
   //      {
   //      if (orig_seq(cnt) != recv_seq(cnt))
   //         no_err++;
   //      }

   //   gamma = pow(P_err, no_err);
   //   gamma *= pow(P_no_err, orig_seq.size() - no_err);

   //   return gamma;

   //   }
   //else
   //   return gamma;

   computer = mychan.get_computer();
   return computer.receive(orig_seq, recv_seq);
   }

template <class sig, class real, class real2>
int conv_modem<sig, real, real2>::get_next_state(int input, int curr_state)
   {
   int state_table_row = bin2int(int2bin(input, k) + int2bin(curr_state,no_states));
   int next_state_loc = k + no_states;
   std::string str_nxt_state = "";
   
   for(int cnt = 0; cnt < no_states;cnt++)
      {
      str_nxt_state += toChar(statetable(state_table_row,next_state_loc));
      next_state_loc++;
      }

   return bin2int(str_nxt_state);
   }

template <class sig, class real, class real2>
unsigned int conv_modem<sig, real, real2>::get_input(unsigned int cur_state, unsigned int next_state)
   {
   for(unsigned int inp = 0; inp < pow(2,k); inp++)
      {
      if(next_state == (unsigned int) get_next_state(inp, cur_state))
         return inp;
      }
   return 0;
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::get_output(int input, int curr_state, array1s_t& output)
   {
   int state_table_row = bin2int(int2bin(input, k) + int2bin(curr_state,no_states));
   int out_loc = k + (no_states * 2);
   
   output.init(n);
   
   //std::string str_output = "";
   for(int cnt = 0; cnt < n;cnt++)
      {
      output(cnt) = statetable(state_table_row, out_loc);
      //std::cout << output(cnt) << std::endl;
      //str_output += toChar(statetable(state_table_row, out_loc));
      out_loc++;
      }
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::get_received(unsigned int b, unsigned int cur_bs, unsigned int next_bs, unsigned int no_del, const array1s_t& rx, array1s_t& recv_codeword)
   {
   recv_codeword.init(next_bs-cur_bs);               

   unsigned int count = 0;

   for(unsigned int i = cur_bs; i < next_bs; i++)
      {
      if(i < (unsigned int) rx.size())
         {
         recv_codeword(count) = rx(i);
         count++;
         }
      else
         {
         break;
         }
      }
   }


// Marker-specific setup functions

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::set_thresholds(const real th_inner,
      const real th_outer)
   {
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   //test_invariant();
   }

// description output

template <class sig, class real, class real2>
std::string conv_modem<sig, real, real2>::description() const
   {
   std::ostringstream sout;
   return sout.str();
   }

// object serialization - saving

template <class sig, class real, class real2>
std::ostream& conv_modem<sig, real, real2>::serialize(std::ostream& sout) const
   {
   sout << "# Type (0 = feedforward, 1 = 1 feedback path, 2 = Multiple feeedback paths)" << std::endl;
   sout << type << std::endl;
   sout << "# no. of inputs" << std::endl;
   sout << k << std::endl;
   sout << "# no. of outputs" << std::endl;
   sout << n << std::endl;
   sout << "# connection matrix for feedforward (octal)" << std::endl;
   sout << ff_octal << std::endl;
   if(type > 0)
      {
      sout << "# connection matrix for feedback (octal)" << std::endl;
      sout << fb_octal << std::endl;
      }
   sout << "# Alphabet size" << std::endl;
   sout << alphabet_size << std::endl;
   sout << "# Block length" << std::endl;
   sout << block_length << std::endl;
   
   sout << "# Maximum Allowable Deletions" << std::endl;
   sout << no_del << std::endl;
   sout << "# Maximum Allowable Insertions" << std::endl;
   sout << no_ins << std::endl;
   sout << "# Maximum Allowable Symbol Shifts" << std::endl;
   sout << rho << std::endl;

   return sout;
   }

// object serialization - loading

template <class sig, class real, class real2>
std::istream& conv_modem<sig, real, real2>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> type >> libbase::verify;
   sin >> libbase::eatcomments >> k >> libbase::verify;
   sin >> libbase::eatcomments >> n >> libbase::verify;
   sin >> libbase::eatcomments;

   feedforward(sin);

   sin >> libbase::eatcomments >> alphabet_size >> libbase::verify;
   sin >> libbase::eatcomments >> block_length >> libbase::verify;

   block_length_w_tail = (ceil((double)(block_length/k)))*n + n*m;

   sin >> libbase::eatcomments >> no_del >> libbase::verify;
   sin >> libbase::eatcomments >> no_ins >> libbase::verify;
   sin >> libbase::eatcomments >> rho >> libbase::verify;
   
   return sin;
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::feedforward(std::istream& sin)
   {
   ffcodebook.init(k,n);
   std::string temp;
      
   std::string* ff_arr = new std::string[n*k];
      
   int * m_arr = new int[k];
   m = 0;
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
         ff_octal += temp + " ";
         ff_arr[cnt] = temp;
         cnt++;
         str_size = oct2bin(temp,0,0).size()-1;
         if(str_size > m_arr[row])
            m_arr[row] = str_size;
         if(m_arr[row] > m)
            m = m_arr[row];
         }
      }

   cnt = 0;
   for(int row = 0; row < k; row++)
      {
      for(int col = 0; col < n; col++)
         {
         /*std::cout << ff_arr[cnt] << std::endl;
         std::cout << oct2bin(ff_arr[cnt], m_arr[row]+1, type) << std::endl;*/
         ffcodebook(row,col) = oct2bin(ff_arr[cnt], m_arr[row]+1, type);
         cnt++;
         //std::cout << ffcodebook(row,col) << std::endl;
         }
      }

   fill_state_diagram_ff(m_arr);
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::fill_state_diagram_ff(int *m_arr)
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
               //bool a = statetable(row,inp_cnt);
               //bool b = toBool(ffcodebook(inp_cnt,out_cnt)[0]);
               temp_out = temp_out ^ (statetable(row,inp_cnt) & toBool(ffcodebook(inp_cnt,out_cnt)[0]));
               //Working effect of shift registers
               for(int conn_cnt = 1; conn_cnt < (m_arr[inp_cnt]+1); conn_cnt++)
                  {
                     //bool c = statetable(row,state_cnt);
                     //bool d = toBool(ffcodebook(inp_cnt,out_cnt)[conn_cnt]);
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

   //disp_statetable();
   }


/*This function converts an integer to a binary String stream.
Input is the integer that needs to be converted
Size is the number of bits that you want the result
Ex. Converting the integer 1, with size 3 will give 001 instead of just 1*/
template <class sig, class real, class real2>
std::string conv_modem<sig, real, real2>::oct2bin(std::string input, int size, int type)
   {
   int div = 0;
   int rem = 0;
   
   //From octal to decimal
   int counter = 0;
   /*Changed here : std::size_t*/
   for(int i = (int)(input.length()-1); i >= 0; i--)
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
	 
         if((int)binary_stream.size() != size && size > 0)
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
int conv_modem<sig, real, real2>::bin2int(std::string binary)
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
std::string conv_modem<sig, real, real2>::int2bin(int input, int size)
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
			
         if((int)binary_stream.size() != size)
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
bool conv_modem<sig, real, real2>::toBool(char const& bit)
   {
     return bit != '0';
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::disp_statetable()
   {
   //system("cls");
   for(int row = 0; row < statetable.size().rows(); row++)
      {
      for(int col = 0; col < statetable.size().cols(); col++)
         {
         std::cout << statetable(row,col) << " ";
         }
      std::cout << std::endl;
      }
   }

template <class sig, class real, class real2>
std::string conv_modem<sig, real, real2>::toString(int number)
   {
   std::stringstream ss;//create a stringstream
   ss << number;
   return ss.str();
   }

template <class sig, class real, class real2>
char conv_modem<sig, real, real2>::toChar(bool bit)
   {
   if(bit)
      return '1';
   else
      return '0';
   }

template <class sig, class real, class real2>
int conv_modem<sig, real, real2>::toInt(bool bit)
   {
   if(bit)
      return 1;
   else
      return 0;
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

/* Serialization string: conv_modem<type,real,real2>
 * where:
 *      type = bool | gf2 | gf4 ...
 *      real = float | double | [logrealfast | mpgnu (CPU only)]
 *      real2 = float | double | [logrealfast | mpgnu (CPU only)]
 */
#define INSTANTIATE3(args) \
      template class conv_modem<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer conv_modem<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockmodem", \
            "conv_modem<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            conv_modem<BOOST_PP_SEQ_ENUM(args)>::create);

#define INSTANTIATE2(r, symbol, reals) \
      INSTANTIATE3( symbol reals )

#define INSTANTIATE1(r, symbol) \
      BOOST_PP_SEQ_FOR_EACH(INSTANTIATE2, symbol, REAL_PAIRS_SEQ)

// NOTE: we *have* to use for-each product here as we cannot nest for-each
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE1, (ALL_SYMBOL_TYPE_SEQ))

} // end namespace
