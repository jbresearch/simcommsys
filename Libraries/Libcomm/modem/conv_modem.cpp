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
   //TODO:Check that the encoder is working as expected
   //Checking that the block lenghts match
   assert(encoded.size() == block_length);
   tx.init(block_length_w_tail);
   encode_data(encoded, tx);
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
   }

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, array1vd_t& ptable)
   {

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

// Marker-specific setup functions

template <class sig, class real, class real2>
void conv_modem<sig, real, real2>::set_thresholds(const real th_inner,
      const real th_outer)
   {
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   test_invariant();
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
