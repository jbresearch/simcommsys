/*!
 * \file
 * $Id: uncoded.cpp 9909 2013-09-23 08:43:23Z jabriffa $
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

#include "conv_codec.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// internal codec operations

template <class dbl>
void conv_codec<dbl>::resetpriors()
   {
   // Allocate space for prior input statistics
   libbase::allocate(rp, This::input_block_size(), This::num_inputs());
   // Initialize
   rp = 1.0;
   }

template <class dbl>
void conv_codec<dbl>::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Copy the input statistics
   rp = ptable;
   }

template <class dbl>
void conv_codec<dbl>::setreceiver(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Copy the output statistics
   R = ptable;
   }

// encoding and decoding functions

template <class dbl>
void conv_codec<dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
   /*std::cout << std::endl;
   std::cout << std::endl;
   std::cout << "Original Data" << std::endl;
   for(int i = 0;i<source.size();i++)
      std::cout << source(i) << " ";*/

   encoded.init(block_length_w_tail);
   encode_data(source, encoded);
   }

template <class dbl>
void conv_codec<dbl>::encode_data(const array1i_t& source, array1i_t& encoded)
   {
   int out_loc = k+(no_states*2);
   int ns_loc = k+no_states;//next state location
   int encoding_counter = 0;
   int tx_cnt = 0;
   int row = 0;
   
   //Resetting encoding steps to 0
   encoding_steps = 0;

   std::string curr_state = "";
   std::string input_and_state = "";
   for(int s = 0; s < no_states;s++)
      curr_state = curr_state + "0";
   
   std::string input = "";
   
   while(encoding_counter < source.size())
      {
      input = "";
      //Getting input
      for(int inp_cnt = 0;inp_cnt < k;inp_cnt++)
         {
         if(encoding_counter > source.size())
            input += "0";
         else
            input += toString(source(encoding_counter));

         encoding_counter++;
         }

      input_and_state = input + curr_state;
      row = bin2int(input_and_state);
      //Encoding
      for(int out_cnt = 0; out_cnt < n; out_cnt++)
         {
         encoded(tx_cnt) = statetable(row, out_loc+out_cnt);
         tx_cnt++;
         }
      
      //Changing current state
      for(int cnt = 0; cnt < no_states; cnt++)
         {
         curr_state[cnt] = toChar(statetable(row,ns_loc+cnt));
         }
      encoding_steps++;
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
         encoded(tx_cnt) = statetable(row, out_loc+out_cnt);
         tx_cnt++;
         }
      
      for(int cnt = 0; cnt < no_states; cnt++)
         {
         curr_state[cnt] = toChar(statetable(row,ns_loc+cnt));
         }
      }
   }

template <class dbl>
void conv_codec<dbl>::softdecode(array1vd_t& ri)
   {
   recv_sequence = R.size();

   libbase::matrix<std::vector<double> > gamma;
   libbase::matrix<double> alpha;
   libbase::matrix<double> beta;
   libbase::matrix<double> output_symbol;
   libbase::matrix<double> output_bit;

   init_matrices(gamma, alpha, beta, output_symbol, output_bit);

   work_gamma(gamma, R);
   work_alpha(gamma, alpha);
   work_beta(gamma, beta);

   decode(gamma, alpha, beta, output_symbol);

   /*Dealing with multiple inputs*/
   multiple_inputs(output_symbol, output_bit);

   /*Normalisation*/
   normalize(output_bit);
   //normalize(output_symbol);

   fill_ptable(ri, output_bit);
   //fill_ptable(ro_var, output_symbol);
  
   /*For testing purposes - BEGIN*/
   //for(int i = 0; i < ri.size(); i++)
   //   std::cout << ri(i) << std::endl;
   /*For testing purposes - END*/

   /*libbase::vector<double> softout;
   softout.init((recv_sequence/n)*k);
   
   work_softout(output_bit, softout);
   
   std::cout << std::endl;
   std::cout << std::endl;
   std::cout << "Decoded Data (prob)" << std::endl;
   for(int i = 0; i < softout.size();i++)
      std::cout << softout(i) << " ";*/
   }

template <class dbl>
void conv_codec<dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   // Determine input-referred statistics
   softdecode(ri);
   ro = R;
   }

template <class dbl>
void conv_codec<dbl>::init_matrices(libbase::matrix<std::vector<double> >& gamma, libbase::matrix<double>& alpha, libbase::matrix<double>& beta, libbase::matrix<double>& output_symbol, libbase::matrix<double>& output_bit)
   {
   init_gamma(gamma);
   init_alpha(alpha);
   init_beta(beta);
   init_output_symbol(output_symbol);
   init_output_bit(output_bit);
   }

template <class dbl>
void conv_codec<dbl>::init_gamma(libbase::matrix<std::vector<double> >& gamma)
   {
   gamma.init(pow(2,no_states), recv_sequence/n);

   for(int col = 0; col < gamma.size().cols(); col++)
      for(int row = 0; row < gamma.size().rows(); row++)
         for(int i = 0; i < pow(2,no_states);i++)
            gamma(row,col).push_back(-1.0);
   }

template <class dbl>
void conv_codec<dbl>::init_alpha(libbase::matrix<double>& alpha)
   {
   alpha.init(pow(2,no_states), recv_sequence/n + 1);
   alpha = 0;
   alpha(0,0) = 1.0;
   }

template <class dbl>
void conv_codec<dbl>::init_beta(libbase::matrix<double>& beta)
   {
   beta.init(pow(2,no_states),recv_sequence/n + 1);
   beta = 0;
   beta(0,beta.size().cols()-1) = 1.0;
   }

template <class dbl>
void conv_codec<dbl>::init_output_symbol(libbase::matrix<double>& output_symbol)
   {
   output_symbol.init(pow(2,k),recv_sequence/n);
   output_symbol = 0;
   }

template <class dbl>   
void conv_codec<dbl>::init_output_bit(libbase::matrix<double>& output_bit)
   {
   output_bit.init(2,(recv_sequence/n)*k);
   output_bit = 0;
   }

template <class dbl>
void conv_codec<dbl>::work_gamma(libbase::matrix<std::vector<double> >& gamma, array1vd_t& recv_ptable)
   {
   int inp_combinations = pow(2,k);
   //double Lc = 5.0;
   int state_table_row = 0;
   int _nxt_state = 0;

   std::vector<int> current_state;
   std::vector<int> next_state;
   bool init_done = false;
   current_state.push_back(0);

   for(int col = 0; col < gamma.size().cols(); col++)
      {
      for(int row = 0; row < pow(2,no_states); row++)
         {
         for(int input = 0; input < inp_combinations; input++)
            {
            _nxt_state = get_next_state(input,row, state_table_row);
            
            if(init_done == false)//Initial special case
               {
               /*Checking if the current state is in the current state vector*/
               if(std::find(current_state.begin(), current_state.end(), row)!=current_state.end())
                  {
                  /*Found*/
                  next_state.push_back(_nxt_state);
                  //gamma(_nxt_state,col)[row] = calc_gamma_AWGN(state_table_row, col, recv, Lc);
                  gamma(_nxt_state,col)[row] = calc_gamma_prob(state_table_row, col, recv_ptable);
                  }
               else
                  {
                  /*Not Found*/
                  gamma(_nxt_state,col)[row] = 0;
                  }
               }
            else if(col >= encoding_steps)//Tailing off
               {
               if(input==0 && std::find(current_state.begin(), current_state.end(), row)!=current_state.end())
                  {
                     //gamma(_nxt_state,col)[row] = calc_gamma_AWGN(state_table_row, col, recv, Lc);
                     gamma(_nxt_state,col)[row] = calc_gamma_prob(state_table_row, col, recv_ptable);
                     next_state.push_back(_nxt_state);
                  }
               else
                  {
                     gamma(_nxt_state,col)[row] = 0;
                  }
               }
            else
               {
               //gamma(_nxt_state,col)[row] = calc_gamma_AWGN(state_table_row, col, recv, Lc);
               gamma(_nxt_state,col)[row] = calc_gamma_prob(state_table_row, col, recv_ptable);
               }
            }
         }
      
      if(init_done == false && next_state.size() == pow(2,no_states))
         {
         init_done = true;
         current_state = next_state;
         next_state.clear();
         }
      else if(init_done == false || col >= encoding_steps)
         {
         current_state = next_state;
         next_state.clear();
         }
      }
   }

template <class dbl>
double conv_codec<dbl>::calc_gamma_prob(int state_table_row, int col, array1vd_t& recv_ptable)
   {
   double temp_gamma = 0.0;
   int out_loc = k+(2*no_states);
   int recv_loc = col * n;

   for(int output = 0; output < n;output++)
      {
      if(output == 0)
         temp_gamma = recv_ptable(recv_loc)(toInt(statetable(state_table_row,out_loc)));
      else
         temp_gamma *= recv_ptable(recv_loc)(toInt(statetable(state_table_row,out_loc)));
      out_loc++;
      recv_loc++;
      }
   return temp_gamma;
   }

template <class dbl>
double conv_codec<dbl>::calc_gamma_AWGN(int state_table_row, int col, double* recv, double Lc)
   {
   double temp_gamma = 0.0;
   int out_loc = k+(2*no_states);
   int recv_loc = col * n;

   for(int output = 0; output < n;output++)
      {
      temp_gamma = temp_gamma + ( toInt(statetable(state_table_row,out_loc)) ) * (recv[recv_loc]);
      out_loc++;
      recv_loc++;
      }
   return exp((Lc/2)*(temp_gamma));
   }

template <class dbl>
void conv_codec<dbl>::work_alpha(libbase::matrix<std::vector<double> >& gamma, libbase::matrix<double>& alpha)
   {
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
                  }
               }
            alpha_total += alpha(row,col);
         }
      for(int r = 0; r < alpha.size().rows();r++)//Normalisation
         {
         alpha(r,col) /= alpha_total;
         }
      }
   }

template <class dbl>
void conv_codec<dbl>::work_beta(libbase::matrix<std::vector<double> >& gamma, libbase::matrix<double>& beta)
   {
   int inp_combinations = pow(2,k);
   int _nxt_state = 0;
   double beta_total = 0.0;

   for(int col = (int)(beta.size().cols()-2); col >= 0; col--)
      {
      beta_total = 0.0;
      for(int row = 0; row < beta.size().rows();row++)
         {
         for(int input = 0; input < inp_combinations; input++)
            {
            _nxt_state = get_next_state(input, row);
            beta(row,col) += beta(_nxt_state,col+1)*gamma(_nxt_state,col)[row];//error here
            }
         beta_total += beta(row,col); 
         }
      for(int r = 0; r < beta.size().rows();r++)//Normalisation
         {
         beta(r,col) /= beta_total;
         }
      }
   }

template <class dbl>
void conv_codec<dbl>::decode(libbase::matrix<std::vector<double> >& gamma, libbase::matrix<double>& alpha, libbase::matrix<double>& beta, libbase::matrix<double>& output_symbol)
   {
   int inp_combinations = pow(2,k);
   int _nxt_state = 0;

   for(int col = 0; col < gamma.size().cols(); col++)
      {
      for(int row = 0; row < alpha.size().rows(); row++)
         {
         for(int input = 0; input < inp_combinations; input++)
            {
            _nxt_state = get_next_state(input, row);

            if(gamma(_nxt_state,col)[row] != -1)
               {
               output_symbol(input,col) += alpha(row,col)*gamma(_nxt_state,col)[row]*beta(_nxt_state,col+1);
               }
            }
         }
      }
   }

template <class dbl>
void conv_codec<dbl>::multiple_inputs(libbase::matrix<double>& output_symbol, libbase::matrix<double>& output_bit)
   {
   std::string binary_input = "";
   
   if(k > 1)
      {
      for(int col = 0; col < output_symbol.size().cols(); col++)
         {
         for(int row = 0; row < output_symbol.size().rows(); row++)
            {
            binary_input = int2bin(row,k);
            for(int str_cnt = 0; str_cnt < (int)binary_input.size(); str_cnt++)
               {
               if(binary_input[str_cnt] == '0')
                  {
                  output_bit(0,(col*k)+str_cnt) += output_symbol(row,col);
                  }
               else
                  {
                  output_bit(1,(col*k)+str_cnt) += output_symbol(row,col);
                  }
               }
            }
         }
      }
   else
      {
      output_bit = output_symbol;
      }
   }

template <class dbl>
void conv_codec<dbl>::fill_ptable(array1vd_t& ptable, libbase::matrix<double>& output_bit)
   {
   //ptable.init(output_bit.size().cols());
   ptable.init(block_length);
   for(int col = 0; col < ptable.size(); col++)
      {
      ptable(col).init(2);
      for(int row = 0; row < ptable(col).size(); row++)
         {
         ptable(col)(row) = output_bit(row,col);
         }
      }
   }

template <class dbl>
void conv_codec<dbl>::work_softout(libbase::matrix<double>& output_bit, libbase::vector<double>& softout)
   {
   for(int cnt = 0; cnt < softout.size(); cnt++)
      softout(cnt) = log(output_bit(1,cnt)/output_bit(0,cnt));
   }

template <class dbl>
void conv_codec<dbl>::normalize(libbase::matrix<double>& mat)
   {
   double norm_total;
   for(int col = 0; col < mat.size().cols();col++)
      {
      norm_total = 0.0;
      for(int row = 0; row < mat.size().rows();row++)
         {
         norm_total += mat(row,col);
         }
      for(int row = 0; row < mat.size().rows();row++)
         {
         mat(row,col) /= norm_total;
         }
      }
   }

template <class dbl>
int conv_codec<dbl>::get_next_state(int input, int curr_state)
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

template <class dbl>
int conv_codec<dbl>::get_next_state(int input, int curr_state, int& state_table_row)
   {
   state_table_row = bin2int(int2bin(input, k) + int2bin(curr_state,no_states));
   int next_state_loc = k + no_states;
   std::string str_nxt_state = "";
   
   for(int cnt = 0; cnt < no_states;cnt++)
      {
      str_nxt_state += toChar(statetable(state_table_row,next_state_loc));
      next_state_loc++;
      }

   return bin2int(str_nxt_state);
   }

// description output
template <class dbl>
std::string conv_codec<dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Uncoded Representation (" << block_length << "x" << alphabet_size << ")";
   return sout.str();
   }

// object serialization - saving
template <class dbl>
std::ostream& conv_codec<dbl>::serialize(std::ostream& sout) const
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
   return sout;
   }

// object serialization - loading
template <class dbl>
std::istream& conv_codec<dbl>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> type >> libbase::verify;
   sin >> libbase::eatcomments >> k >> libbase::verify;
   sin >> libbase::eatcomments >> n >> libbase::verify;
   sin >> libbase::eatcomments;

   if(type == 0)
      feedforward(sin);
   else if(type > 0)
      feedback(sin);

   sin >> libbase::eatcomments >> alphabet_size >> libbase::verify;
   sin >> libbase::eatcomments >> block_length >> libbase::verify;

   block_length_w_tail = (ceil((double)(block_length/k)))*n + n*m;

   return sin;
   }

template <class dbl>
void conv_codec<dbl>::feedforward(std::istream& sin)
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

template <class dbl>
void conv_codec<dbl>::feedback(std::istream& sin)
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
         ff_octal += temp + " ";
         ff_arr[cnt] = temp;
         if((int)oct2bin(temp,0,0).size() > m)
            m = oct2bin(temp,0,0).size();
      }

   sin >> libbase::eatcomments;

   for(int cnt = 0; cnt < (k*n); cnt++)
      {
         sin >> temp;
         fb_octal += temp + " ";
         fb_arr[cnt] = temp;
         if((int)oct2bin(temp,0,0).size() > m)
            m = oct2bin(temp,0,0).size();
      }

   int counter = 0;
   for(int row = 0; row < k; row++)
      {
      for(int col = 0; col < n; col++)
         {            
         ffcodebook(row,col) = oct2bin(ff_arr[counter],m, type);
         counter++;
         //std::cout << ffcodebook(row,col) << std::endl;
         }
      }

   counter = 0;
   for(int row = 0; row < k; row++)
      {
      for(int col = 0; col < n; col++)
         {
         fbcodebook(row,col) = oct2bin(fb_arr[counter],m, type);
         counter++;
         //std::cout << fbcodebook(row,col) << std::endl;
         }
      }
   delete[] fb_arr;
   delete[] ff_arr;
   m--;
   fill_state_diagram_fb();
   }

template <class dbl>
void conv_codec<dbl>::fill_state_diagram_ff(int *m_arr)
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



template <class dbl>
void conv_codec<dbl>::fill_state_diagram_fb()
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
                     //bool x = toBool(fbcodebook(input_col,out_cnt)[i]);
                     //bool y = statetable(row,i);
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
                     //bool x = toBool(ffcodebook(input_col,out_cnt)[i]);
                     //bool y = feedback;
                     output = output ^ (toBool(ffcodebook(input_col,out_cnt)[i]) & feedback);
                     }
                  else
                     {
                     //bool x = toBool(ffcodebook(input_col,out_cnt)[i]);
                     //bool y = statetable(row,(k+i)-1);
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
      bool output = 0;
      bool temp_out = 0;

      int fb_loc = 0;

      for(int row = 0; row < no_rows; row++)
         {
         /*Resetting every iteration*/
         feedback = 0;
         output_col = k+2*m;//location of first output index
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
                        //bool x = toBool(ffcodebook(inp_cnt,out_cnt)[i]);
                        //bool y = statetable(row,state_loc);
                        temp_out = temp_out ^ (toBool(ffcodebook(inp_cnt,out_cnt)[i]) & statetable(row,state_loc));
                        state_loc--;   
                        }
                     else //Getting the relevant input
                        {
                        //bool x = toBool(ffcodebook(inp_cnt,out_cnt)[i]);
                        //bool y = statetable(row,inp_cnt);
                        temp_out = temp_out ^ (toBool(ffcodebook(inp_cnt,out_cnt)[i]) & statetable(row,inp_cnt));
                        }
                     }
                  output = output ^ temp_out;
                  }
               else//has feedback
                  {
                  fb_loc = out_cnt;
                  //bool x = toBool(ffcodebook(inp_cnt,out_cnt)[m]);
                  //bool y = statetable(row,inp_cnt);
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
                  //bool x = toBool(ffcodebook(inp_cnt,fb_loc)[i]);
                  //bool y = statetable(row,inp_cnt);
                  state = state ^ ( toBool(ffcodebook(inp_cnt,fb_loc)[i]) & statetable(row,inp_cnt));
                  }
               //bool t = toBool(fbcodebook(0,fb_loc)[i]);
               state = state ^ ( toBool(fbcodebook(0,fb_loc)[i]) & feedback);
               //check whether it has previous state
               if((state_loc-1) >= (k+m))
                  {
                  //bool a = statetable(row, (state_loc-m-1));
                  state = state ^ statetable(row, (state_loc-m-1));
                  }
               statetable(row,state_loc) = state;
               state = 0;
               state_loc++;
               }
         }
      }

   //disp_statetable();

   }

/*This function converts an integer to a binary String stream.
Input is the integer that needs to be converted
Size is the number of bits that you want the result
Ex. Converting the integer 1, with size 3 will give 001 instead of just 1*/
template <class dbl>
std::string conv_codec<dbl>::oct2bin(std::string input, int size, int type)
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

template <class dbl>
int conv_codec<dbl>::bin2int(std::string binary)
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

template <class dbl>
std::string conv_codec<dbl>::int2bin(int input, int size)
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

template <class dbl>
bool conv_codec<dbl>::toBool(char const& bit)
   {
     return bit != '0';
   }

template <class dbl>
void conv_codec<dbl>::disp_statetable()
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

template <class dbl>
std::string conv_codec<dbl>::toString(int number)
   {
   std::stringstream ss;//create a stringstream
   ss << number;
   return ss.str();
   }

template <class dbl>
char conv_codec<dbl>::toChar(bool bit)
   {
   if(bit)
      return '1';
   else
      return '0';
   }

template <class dbl>
int conv_codec<dbl>::toInt(bool bit)
   {
   if(bit)
      return 1;
   else
      return 0;
   }

//template <class dbl>
//void conv_codec<dbl>::settoval(libbase::matrix<double>& mat, double value)
//   {
//   for(int row = 0; row < mat.size().rows(); row++)
//      for(int col = 0; col < mat.size().cols(); col++)
//         mat(row,col) = value;
//   }


} // end namespace

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)

/* Serialization string: uncoded<real>
 * where:
 *      real = float | double | mpreal | mpgnu | logreal | logrealfast
 */
#define INSTANTIATE(r, x, type) \
      template class conv_codec<type>; \
      template <> \
      const serializer conv_codec<type>::shelper( \
            "codec", \
            "conv_codec<" BOOST_PP_STRINGIZE(type) ">", \
            conv_codec<type>::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
