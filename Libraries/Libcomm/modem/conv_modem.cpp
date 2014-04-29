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

#include "randgen.h"
#include "truerand.h"
//#include "random.h"

#include <math.h>
/*For overflow*/
//#include <cfenv>
#include <iostream>
//#include <stdlib.h>     /* srand, rand */
//#include <time.h>

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

typedef long double dbl;

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
   assert(encoded.size() == block_length);
   tx.init(block_length_w_tail);
   encode_data(encoded, tx);

   ////std::cout << "Encoded: " << std::endl;

   ////for (int i = 0; i < encoded.size(); i++)
   ////   {
   ////   std::cout << encoded(i) << " ";
   ////   }

   //std::cout << std::endl;
   //std::cout << "Tx before: " << std::endl;

   //for (int i = 0; i < tx.size(); i++)
   //   {
   //   std::cout << tx(i) << " ";
   //   }

   if (add_rand_seq == 1)
      {
      create_random();
      add_random(tx);
      }
   
   //std::cout << std::endl;
   //std::cout << "Random Sequence: " << std::endl;

   //for (int i = 0; i < random_sequence.size(); i++)
   //   {
   //   std::cout << random_sequence[i] << " ";
   //   }

   //std::cout << std::endl;
   //std::cout << "Tx: " << std::endl;

   //for (int i = 0; i < tx.size(); i++)
   //   {
   //   std::cout << tx(i) << " ";
   //   }

   //std::cout << std::endl;
   //std::cout << std::endl;
   //disp_statetable();

   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::encode_data(const array1i_t& encoded, array1s_t& tx)
   {
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

   if (no_states > 0)//1 state change
      {
      //Tailing off
      for (int tailoff_cnt = 0; tailoff_cnt < m; tailoff_cnt++)
         {
         input = "";
         for (int inp_cnt = 0; inp_cnt < k; inp_cnt++)
            {
            input += "0";
            }

         input_and_state = input + curr_state;
         row = bin2int(input_and_state);
         for (int out_cnt = 0; out_cnt < n; out_cnt++)
            {
            tx(tx_cnt) = statetable(row, out_loc + out_cnt);
            tx_cnt++;
            }

         for (int cnt = 0; cnt < no_states; cnt++)
            {
            curr_state[cnt] = toChar(statetable(row, ns_loc + cnt));
            }
         }
      }
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::add_random(array1s_t& tx)
   {
   for (int i = 0; i < tx.size(); i++)
      {
      //std::cout << tx(i) << " + " << random_sequence[i] << " = ";
      tx(i) = (bool) tx(i) ^ (bool) random_sequence[i];
      //std::cout << tx(i) << std::endl;
      }
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::create_random()
   {
   random_sequence.resize(block_length_w_tail);
   if (add_rand_seq == 0)
      {
      std::fill(random_sequence.begin(), random_sequence.end(), 0);
      }
   else
      {
      
      //random_sequence[0] = 1;
      //random_sequence[1] = 0;
      //random_sequence[2] = 0;
      //random_sequence[3] = 1;
      //random_sequence[4] = 0;
      //random_sequence[5] = 0;
      //random_sequence[6] = 1;
      //random_sequence[7] = 1;
      //random_sequence[8] = 0;
      //random_sequence[9] = 0;
      libbase::truerand trng;

      for (int i = 0; i < block_length_w_tail; i++)
         {
         if (trng.fval_closed() > 0.5)
            random_sequence[i] = 1;
         else
            random_sequence[i] = 0;
         }
      }
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::dodemodulate(const channel<sig>& chan, const array1s_t& rx, array1vd_t& ptable)
   {

   //std::cout << std::endl;
   //std::cout << "(recv) Random Sequence: " << std::endl;

   //for (int i = 0; i < random_sequence.size(); i++)
   //   {
   //   std::cout << random_sequence[i] << " ";
   //   }

   //std::cout << std::endl;
   //std::cout << "Rx: " << std::endl;

   //for (int i = 0; i < rx.size(); i++)
   //   {
   //   std::cout << rx(i) << " ";
   //   }
   
   gamma_storage.clear();
   vector_3d().swap(gamma_storage);
   gamma_storage.resize(pow(2,no_states));

   unsigned int no_insdels=0;

   unsigned int b_size = block_length + m + 1;

   unsigned int num_states = pow(2, no_states);

   vector<b_storage> b_vector(b_size, b_storage(num_states));

   /*Channel related initialisations - BEGIN*/
   mychan = dynamic_cast<const qids<sig, real2>&> (chan);
   mychan.set_blocksize(2);

   vector <dynamic_symbshift> vec_symbshift;// (b_size);
   
   if (dynamic_limit > 0)
      {
      vec_symbshift.resize(b_size);
      int min, max;
      
      mychan.set_pr(dynamic_limit);
      computer = mychan.get_computer();

      if (old_pi != mychan.get_pi() || old_pd != mychan.get_pd())
         {
         /*Setting up inital values for lambda - begin*/
         mychan.compute_limits(n, dynamic_limit, min, max);
         no_ins = max;
         no_del = abs(min);
         /*Setting up inital values for lambda - end*/

         /*Setting up dynamic rho vector - begin*/
         for (unsigned int i = 1; i < b_size; i++)
            {
            mychan.compute_limits(i*n, dynamic_limit, min, max);
            min = ceil((double) abs(min) / (double)n);
            max = ceil((double) max / (double) n);
            vec_symbshift[i].setminmax(min, max);
            }
         /*Setting up dynamic rho vector - end*/
         }
      }

   /*Channel related initialisations - END*/
   
   no_insdels = no_del + no_ins + 1;

   /*Setting up the first alpha value - BEGIN*/
   b_vector[0].setmin_bs(0);
   //b_vector[0].state_bs_vector.resize(1);
   b_vector[0].state_bs_vector[0].push_back(state_bs_storage(1));
   /*Setting up the first alpha value - END*/

   unsigned int inp_combinations = pow(2,k);
   unsigned int next_state = 0;

   array1s_t orig_codeword;
   array1s_t recv_codeword;               

   dbl gamma = 0.0;
   dbl alpha = 0.0;
   dbl beta = 0.0;
   dbl alpha_total = 0.0;
   dbl beta_total = 0.0;
   unsigned int num_bs = 0;

   unsigned int cur_bs = 0;
   int next_bs = 0;

   unsigned int norm_b = 0;

   int drift = 0;
   int symb_shift = 0;//the current number of symbol shifts

   unsigned int recv_size = rx.size();

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

      
      if (no_states > 0)//1 state change
         {
         if (b >= (unsigned int)block_length) //this is tailing so input is always 0
            inp_combinations = 1;
         }
               
      if (no_states > 0)
         {
         if ((b + 1) == b_size)
            b_vector[b + 1].setmin_bs(recv_size);
         else
            b_vector[b + 1].setmin_bs(b_vector[b].getmin_bs() + n - no_del);
         }
      else
         {
            b_vector[b + 1].setmin_bs(b_vector[b].getmin_bs() + n - no_del);
         }
      
      //For all the number of states
      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[b].state_bs_vector[cur_state].size();
         
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            cur_bs = b_vector[b].getmin_bs() + cnt_bs;
            
               for (unsigned int input = 0; input < inp_combinations; input++)
                  {
                  next_state = get_next_state(input, cur_state);
                  next_bs = cur_bs + n - no_del;//setting up the initial point of next_bs
                     
                  get_output(input, cur_state, orig_codeword, b);

                  //std::cout << std::endl;
                  //std::cout << "Output at (" << input << "," << cur_state << ") b = " << b << std::endl;
                  //for (int i = 0; i < orig_codeword.size(); i++)
                  //   {
                  //   std::cout << orig_codeword(i) << " ";
                  //   }
                  //std::cout << std::endl;

                  for (unsigned int cnt_next_bs = 0; cnt_next_bs < no_insdels; cnt_next_bs++)
                     {
                     if (next_bs >= (int) cur_bs)
                        {
                        if (b_vector[b + 1].getmin_bs() < 0)
                           {
                           b_vector[b + 1].setmin_bs(next_bs);
                           }
                           
                        /*Calculating the current drift - BEGIN*/
                        drift = next_bs - (b + 1)*n;
                        symb_shift = ceil(dbl(abs(drift)) / dbl(n));

                        if (dynamic_limit > 0)
                           {
                           if (drift < 0)
                              rho = vec_symbshift[b + 1].getmin();
                           else
                              rho = vec_symbshift[b + 1].getmax();
                           }
                        /*Calculating the current drift - END*/

                        if ((((b + 1) == b_size) && next_bs == (int) recv_size) || (((b + 1) < b_size) && (symb_shift <= rho)))
                           {
                           get_received(b, cur_bs, next_bs, no_del, rx, recv_codeword);
                           
                           gamma = work_gamma(orig_codeword, recv_codeword);//1 state change
                           
                           /*std::cout << std::endl;
                           std::cout << "Received at (" << cur_bs << "," << next_bs << ") b = " << b  << " gamma = " << gamma << std::endl;
                           for (int i = 0; i < recv_codeword.size(); i++)
                              {
                              std::cout << recv_codeword(i) << " ";
                              }
                           std::cout << std::endl;*/

                           //gamma = get_gamma(cur_state, cur_bs, next_state, next_bs, orig_codeword, recv_codeword);

                           //Work alpha
                           unsigned int st_cur_bs = (cur_bs - b_vector[b].getmin_bs());//the actual store location for current bs
                           unsigned int st_nxt_bs = (next_bs - b_vector[b + 1].getmin_bs());//the actual store location for next bs
                           //For release
                           alpha = gamma * b_vector[b].state_bs_vector[cur_state][st_cur_bs].getalpha();
                           alpha_total += alpha;

                           //storing gamma
                           /*Check whether bit shift location is already available - Begin*/
                           if (b_vector[b + 1].state_bs_vector[next_state].size() < (st_nxt_bs + 1))
                              b_vector[b + 1].state_bs_vector[next_state].resize(st_nxt_bs + 1);
                           /*Check whether bit shift location is already available - End*/

                           b_vector[b + 1].state_bs_vector[next_state][st_nxt_bs].gamma.push_back(Gamma_Storage(cur_state, st_cur_bs, gamma));
                           //storing alpha
                           b_vector[b + 1].state_bs_vector[next_state][st_nxt_bs].setalpha(alpha);

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
                     else
                        {
                        next_bs++;
                        }
                     }
                  }
            }
         }

      //Normalisation
      norm_b = b+1;

      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
         
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].normalpha(alpha_total);
            }
         }
      }

   /*Setting up the first beta value - BEGIN*/
   b_vector[b_size].state_bs_vector[0][0].setbeta(1);
   /*Setting up the first beta value - END*/

   unsigned int size_gamma = 0;

   unsigned int prev_bs = 0;
   unsigned int prev_state = 0;
   
   std::vector< std::vector<dbl> > vec_tmp_output;
   vec_tmp_output.resize(pow(2,k));

   dbl out_summation = 0.0;
   dbl temp_out = 0.0;

   int counter = 0;//1 state change

   for(unsigned int b = b_size; b > 0; b--)
      {
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
               /*Inserting output values for normalisation - BEGIN*/
               temp_out = alpha * gamma * beta;
               if (no_states == 0)//1 state change
                  {
                  vec_tmp_output[counter].push_back(temp_out);
                  counter++;
                  if (counter == 2) counter = 0;
                  }
               else
                  {
                  vec_tmp_output[get_input(prev_state, cur_state)].push_back(temp_out);
                  //outtable(b - 1)(get_input(prev_state, cur_state)) += (alpha * gamma * beta);
                  }
               //vec_tmp_output[get_input(prev_state, cur_state)].push_back(temp_out);
               out_summation += temp_out;
               /*Inserting output values for normalisation - END*/

               //Working out next beta
               beta = beta * gamma;
               beta_total += beta;
               b_vector[b-1].state_bs_vector[prev_state][prev_bs].setbeta(beta);
               }
            }
         }
      assert(out_summation > 0);

      transform(vec_tmp_output[0].begin(), vec_tmp_output[0].end(), vec_tmp_output[0].begin(), bind2nd( divides<dbl>(), out_summation));
      transform(vec_tmp_output[1].begin(), vec_tmp_output[1].end(), vec_tmp_output[1].begin(), bind2nd( divides<dbl>(), out_summation));
      
      outtable(b-1)(0) = std::accumulate(vec_tmp_output[0].begin(), vec_tmp_output[0].end(), 0.0);
      outtable(b-1)(1) = std::accumulate(vec_tmp_output[1].begin(), vec_tmp_output[1].end(), 0.0);

      vec_tmp_output[0].clear();
      vec_tmp_output[1].clear();

      //Normalisation
      norm_b = b-1;

      for(unsigned int cur_state = 0; cur_state < num_states; cur_state++)
         {
         num_bs = b_vector[norm_b].state_bs_vector[cur_state].size();
         //For all the number of bitshifts available
         for(unsigned int cnt_bs = 0; cnt_bs < num_bs; cnt_bs++)
            {
            b_vector[norm_b].state_bs_vector[cur_state][cnt_bs].normbeta(beta_total);
            }
         }
      }
   ptable = outtable.extract(0,block_length);

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

   //std::cin.get();
   //system("cls");

   vec_tmp_output.clear();
   vector< vector<dbl> >().swap(vec_tmp_output);
   
   b_vector.clear();
   vector<b_storage>().swap(b_vector);
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
dbl conv_modem<sig, real, real2>::get_gamma(unsigned int cur_state, unsigned int cur_bs, unsigned int next_state, unsigned int next_bs, array1s_t& orig_seq, array1s_t& recv_seq)
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
   dbl gamma = work_gamma(orig_seq, recv_seq);
   gamma_storage[cur_state][cur_bs].push_back(Gamma_Storage(next_state, next_bs, gamma));

   return gamma;
   }

template <class sig, class real, class real2>
dbl conv_modem<sig, real, real2>::work_gamma(array1s_t& orig_seq, array1s_t& recv_seq)
   {
   double pi, pd;
   
   //(0 = WLD, 1 = Uleven, 2 = Receive Function, 3 = Hamming Distance)
   switch (gamma_calc)
      {
      case 0:
         return WLD(orig_seq, recv_seq);
         break;
      case 1:
         pi = mychan.get_pi();
         pd = mychan.get_pd();
         return uleven_low_soft(orig_seq, recv_seq, mychan.get_ps(), pi, pd, (1 - pi - pd)) * 0.5;
         break;
      case 2:
         return computer.receive(orig_seq, recv_seq);
         break;
      case 3:
         return Hamming(orig_seq, recv_seq);
         break;
      }
   return 0.0;
   }

/*Levenshtein Distance*/
template <class sig, class real, class real2>
int conv_modem<sig, real, real2>::sleven(std::string string1, std::string string2, int sub, int ins, int del)
   {
   int i, j,
      *dist1,
      *dist2,
      *logic1,
      *logic2,
      //*swap_temp,
      cost,
      newd,
      distance;


   int len1 = string1.size();
   int len2 = string2.size();

   /*Allocate memory to store two columns in the distance matrix */
   dist1 = (int *)malloc(sizeof(*dist1) * (size_t)(len2 + 1));
   dist2 = (int *)malloc(sizeof(*dist2) * (size_t)(len2 + 1));


   /*Initialise the logical pointers to the two columns, these would be swapped
   as we go along so as to maintain always the last two columns */
   logic1 = &dist1[0];
   logic2 = &dist2[0];

   /*Initialise the first column to all insertions */
   logic1[0] = 0;
   for (j = 1; j <= len2; j++)
      logic1[j] = logic1[j - 1] + ins;



   /*Do for all columns */
   for (i = 0; i<len1; i++)
      {              /*Initialise always the first row to all deletions */
      logic2[0] = logic1[0] + del;


      for (j = 0; j<len2; j++)
         {
         /*Determine if this is an insertion, deletion or a possible
         substitution by choosing the minimum*/
         if (string1[i] == string2[j])
            cost = 0;
         else
            cost = sub;


         /*Try substitution*/
         cost += logic1[j];



         /*Try insertion*/
         newd = logic2[j] + ins;
         if (newd < cost)
            cost = newd;



         /*Try deletion*/
         newd = logic1[j + 1] + del;
         if (newd < cost)
            cost = newd;


         logic2[j + 1] = cost;

         }

      /*Make the last calculated column to be logic1*/
      swap(logic1, logic2);
      }


   distance = logic1[len2];
   free(dist1);
   free(dist2);

   return distance;

   }

template <class sig, class real, class real2>
double conv_modem<sig, real, real2>::uleven_low_soft(array1s_t& orig_seq, array1s_t& recv_seq, double sub, double ins, double del, double tx)
   {
   int reflen = orig_seq.size();
   int length = recv_seq.size();
   
   int i, j;
   double  *logic1,
      *logic2,
      *swap_temp,
      cost_nosub = tx * (1 - sub),
      cost_sub = tx * sub;

   //Adjust ins to take into account that an insertion of a 0 or a 1 are equi-probable (this is
   //the probability of having a '0' inserted or a '1' inserted
   ins *= 0.5;


   //Pre-extract the individual bits of the two words for faster computation
   //Note that these are still retained in the msb
   for (i = 0; i < orig_seq.size(); i++)
      bits1[i] = orig_seq(i);

   for (i = 0; i < recv_seq.size(); i++)
      bits2[i] = recv_seq(i);

   /*Initialise the logical pointers to the two columns, these would be swapped
   as we go along so as to maintain always the last two columns */
   logic1 = dist1;
   logic2 = dist2;

   /*Initialise the first column to all insertions */
   logic1[0] = 1.0;
   for (j = 1; j <= length; j++)
      {
      logic1[j] = logic1[j - 1] * ins;
      }

   /*Do for all columns, except for the last one which is considered separately since no insertions can occur in the last column */
   for (i = 0; i<reflen - 1; i++)
      {   /*Initialise always the first row to all deletions */
      logic2[0] = logic1[0] * del;

      for (j = 0; j<length; j++)
         {

         //Determine the soft Levenshtein distance
         //this is the summation of all the probabilities of all possible
         //paths within the lattice.
         //Probabilities along one path are multiplied.
         //Probabilities of different paths are added.
         logic2[j + 1] = logic1[j] * (bits1[i] == bits2[j] ? cost_nosub : cost_sub) + logic2[j] * ins + logic1[j + 1] * del;
         }

      /*Make the last calculated column to be logic1*/
      swap_ls(logic1, logic2);
      }

   //Note that insertions in the last codeword bit are not being considered.
   //We are therefore doing the relative calculations separately to increase
   //speed
   logic2[0] = logic1[0] * del;
   for (j = 0; j<length; j++)
      {
      logic2[j + 1] = logic1[j] * (bits1[i] == bits2[j] ? cost_nosub : cost_sub) + logic1[j + 1] * del;
      }
   
   return logic2[length];
   }

template <class sig, class real, class real2>
dbl conv_modem<sig, real, real2>::WLD(array1s_t& orig_seq, array1s_t& recv_seq)
   {

   //orig_seq.init(3);
   //recv_seq.init(4);

   //orig_seq(0) = 1;
   //orig_seq(1) = 0;
   //orig_seq(2) = 1;

   //recv_seq(0) = 0;
   //recv_seq(1) = 1;
   //recv_seq(2) = 0;
   //recv_seq(3) = 1;

   double Pi = mychan.get_pi();
   double Pd = mychan.get_pd();
   double Ps = mychan.get_ps();
   double Pt = 1 - Pi - Pd;

   double Wi = log10(Pi)*-1;
   double Wd = log10(Pd/(Pt*(1-Ps)))*-1;
   double Ws = log10(Ps / (1 - Ps))*-1;

   int N_i,N_d,N_s;

   int col, row;

   N_i = N_d = N_s = 0;

   //Wi = Wd = Ws = 1.0;

   double cost_sub, cost_del, cost_ins;

   for (col = 1; col < orig_seq.size()+1; col++)
      {
      for (row = 1; row < recv_seq.size()+1; row++)
         {
         if (orig_seq(col-1) == recv_seq(row-1))//If no error get the diagonal value
            {
            WLD_vector[row][col] = WLD_vector[row - 1][col - 1];
            }
         else
            {
            cost_sub = WLD_vector[row - 1][col - 1] + Ws;
            cost_del = WLD_vector[row][col - 1] + Wd;
            cost_ins = WLD_vector[row - 1][col] + Wi;
	    
            WLD_vector[row][col] = std::min(std::min(cost_sub,cost_del),cost_ins);
            //WLD_vector[row][col] = std::min({ cost_sub, cost_del, cost_ins });
            }
         }
      }

   col = orig_seq.size();
   row = recv_seq.size();

   double top, left, diagonal,current;
   int row_1, col_1;

   while (!(col == 0 && row == 0))
      {
      current = WLD_vector[row][col];

      row_1 = row - 1;
      col_1 = col - 1;

      if (row_1 < 0)
         top = 10000;
      else
         top = WLD_vector[row_1][col];
      
      if (col_1 < 0)
         left = 10000;
      else
         left = WLD_vector[row][col_1];

      if (row_1 < 0 || col_1 < 0)
         diagonal = 10000;
      else
         diagonal = WLD_vector[row_1][col_1];

      if (top < left && top < diagonal)//Insertion
         {
         if (current != WLD_vector[--row][col])
            N_i++;
         }
      else if (left < top && left < diagonal)//Deletion
         {
         if (current != WLD_vector[row][--col])
            N_d++;
         }
      else//Substitution
         {
         if (current != WLD_vector[--row][--col])
            N_s++;
         }
      }

   //double gamma = pow(Pi, N_i) * pow(Pd, N_d) * pow((Pt*Ps), N_s) * pow((Pt*(1-Ps)), (n-N_d-N_s));

   return pow(Pi, N_i) * pow(Pd, N_d) * pow((Pt*Ps), N_s) * pow((Pt*(1 - Ps)), (n - N_d - N_s));;
   }

template <class sig, class real, class real2>
dbl conv_modem<sig, real, real2>::Hamming(array1s_t& orig_seq, array1s_t& recv_seq)
   {
   double P_err = mychan.get_ps();
   double P_no_err = 1 - P_err;

   int no_err = 0;

   double gamma = 0.0;

   if (orig_seq.size() == recv_seq.size())
      {
      //Calculating the Hamming distance
      for (int cnt = 0; cnt < orig_seq.size(); cnt++)
         {
         if (orig_seq(cnt) != recv_seq(cnt))
            no_err++;
         }

      gamma = pow(P_err, no_err);
      gamma *= pow(P_no_err, orig_seq.size() - no_err);

      return gamma;
      }
   else
      return gamma;
   }


template <class sig, class real, class real2>
int conv_modem<sig, real, real2>::get_next_state(int input, int curr_state)
   {
   return int_statetable[input][curr_state].get_next_state();
   /*int state_table_row = bin2int(int2bin(input, k) + int2bin(curr_state,no_states));
   int next_state_loc = k + no_states;
   std::string str_nxt_state = "";
   
   for(int cnt = 0; cnt < no_states;cnt++)
      {
      str_nxt_state += toChar(statetable(state_table_row,next_state_loc));
      next_state_loc++;
      }

   return bin2int(str_nxt_state);*/
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
void conv_modem<sig, real, real2>::get_output(int input, int curr_state, array1s_t& output, unsigned int b)
   {
   output.init(n);

   int _output = int_statetable[input][curr_state].get_output();

   /*std::cout << std::endl;
   std::cout << "input = " << input << " current state = " << curr_state << " get_output b = " << b << std::endl;*/

   for (int cnt = n - 1; cnt >= 0; cnt--)
      {
      output(cnt) = _output & 1;
      _output = _output >> 1;
      }

   //for (int i = 0; i < output.size(); i++)
   //   std::cout << output(i) << " ";

   //std::cout << std::endl;
   //std::cout << "get_output with added random sequence" << std::endl;
   if (add_rand_seq == 1)
      {
      int start_loc = b*n;
      for (int i = 0; i < n; i++)
         {
         output(i) = (bool)output(i) ^ random_sequence[start_loc];
         //std::cout << output(i) << " ";
         start_loc++;
         }
      }

   //std::cout << std::endl;
   /*int state_table_row = bin2int(int2bin(input, k) + int2bin(curr_state,no_states));
   int out_loc = k + (no_states * 2);
   
   t_output.init(n);
   
   for(int cnt = 0; cnt < n;cnt++)
      {
      t_output(cnt) = statetable(state_table_row, out_loc);
      out_loc++;
      }*/
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::get_received(unsigned int b, unsigned int cur_bs, unsigned int next_bs, unsigned int no_del, const array1s_t& rx, array1s_t& recv_codeword)
   {
   
   //if (cur_bs > rx.size())
   //   {
   //   recv_codeword.init(0);
   //   break;
   //   }

   //if (next_bs > (unsigned int) (rx.size()+1))
   //   recv_codeword.init(rx.size() - cur_bs);
   //else
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
   sout << "# Gamma Calculation(0 = WLD, 1 = Uleven, 2 = Receive Function)" << std::endl;
   sout << gamma_calc << std::endl;

   sout << "# Dynamic Deletions/Insertions (0 = no takes fixed values, any other value is the probability of channel event outside chosen limits)" << std::endl;
   sout << dynamic_limit << std::endl;
   sout << "# Maximum Allowable Deletions" << std::endl;
   sout << no_del << std::endl;
   sout << "# Maximum Allowable Insertions" << std::endl;
   sout << no_ins << std::endl;
   sout << "# Maximum Allowable Symbol Shifts" << std::endl;
   sout << rho << std::endl;
   
   sout << "# Addition of random sequence(0 = no, 1 = yes)" << std::endl;
   sout << add_rand_seq << std::endl;
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

   sin >> libbase::eatcomments >> gamma_calc >> libbase::verify;

   block_length_w_tail = (ceil((double)(block_length/k)))*n + n*m;

   sin >> libbase::eatcomments >> dynamic_limit >> libbase::verify;
   sin >> libbase::eatcomments >> no_del >> libbase::verify;
   sin >> libbase::eatcomments >> no_ins >> libbase::verify;
   sin >> libbase::eatcomments >> rho >> libbase::verify;
   
   sin >> libbase::eatcomments >> add_rand_seq >> libbase::verify;

   /*Filling int_statetable*/
   fill_intstatetable();

   /*Generating random sequence*/
   //create_random();

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

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::fill_intstatetable(void)
   {
   if (no_states == 0)
      {
      unsigned int rows = 2;
      unsigned int cols = 1;

      int_statetable.resize(rows, vector<state_output>(cols));

      int_statetable[0][0].set_next_state(0);
      int_statetable[0][0].set_output(0);

      int_statetable[1][0].set_next_state(0);
      int_statetable[1][0].set_output(7);
      }
   else
      {
      unsigned int rows = pow(2, k);
      unsigned int cols = pow(2, no_states);

      int_statetable.resize(rows, vector<state_output>(cols));

      unsigned int out_loc = k + (no_states * 2);
      unsigned int cur_state_loc = k;
      unsigned int next_state_loc = k + no_states;

      std::string input = "";
      std::string cur_state = "";
      std::string next_state = "";
      std::string output = "";

      for (int state_table_row = 0; state_table_row < statetable.size().rows(); state_table_row++)
         {
         /*Getting Input*/
         for (int cnt = 0; cnt < k; cnt++)
            input += toChar(statetable(state_table_row, cnt));
         /*Getting Current State*/
         for (int cnt = 0; cnt < no_states; cnt++)
            cur_state += toChar(statetable(state_table_row, cnt + cur_state_loc));
         /*Getting Next State*/
         for (int cnt = 0; cnt < no_states; cnt++)
            next_state += toChar(statetable(state_table_row, cnt + next_state_loc));
         /*Getting Output*/
         for (int cnt = 0; cnt < n; cnt++)
            output += toChar(statetable(state_table_row, cnt + out_loc));

         int_statetable[bin2int(input)][bin2int(cur_state)].set_next_state(bin2int(next_state));
         int_statetable[bin2int(input)][bin2int(cur_state)].set_output(bin2int(output));

         /*Resetting Variables*/
         input = "";
         cur_state = "";
         next_state = "";
         output = "";
         }
      }
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
        unsigned int str_size = binary.size() - 1;

        for (unsigned int i = 0; i < str_size; i++)
           {
           result = (result | (binary[i]-'0')) << 1;
           }

        result = result | (binary[str_size] - '0');

        return result;

	//reverse(binary.begin(), binary.end());

	//for(unsigned int i = 0; i < binary.size();i++)
	//{
	//	if(binary[i] == '1')
	//	{
	//		result = result + (int) pow(2,i);
	//	}
	//}
	
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
