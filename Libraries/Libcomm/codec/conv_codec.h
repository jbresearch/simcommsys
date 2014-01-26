/*!
 * \file
 * $Id: uncoded.h 9909 2013-09-23 08:43:23Z jabriffa $
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

#ifndef __conv_codec_h
#define __conv_codec_h

#include "config.h"

#include "codec_softout.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Uncoded transmission.
 * \author  Johann Briffa
 * $Id: uncoded.h 9909 2013-09-23 08:43:23Z jabriffa $
 *
 * This class represents the simplest possible encoding, where the output is
 * simply a copy of the input. Equivalently, at the receiving end, the
 * decoder soft-output is simply a copy of its soft-input.
 */

//template <class dbl = double>
template <class dbl>
class conv_codec : public codec_softout<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef conv_codec<dbl> This;
   typedef codec_softout<libbase::vector, dbl> Base;
   typedef dbl t_dbl;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   array1vd_t rp; //!< Intrinsic source statistics
   array1vd_t R; //!< Intrinsic output statistics

   /*Conv Codes parameters - BEGIN*/
   int type;
   int alphabet_size; //!< Alphabet size (input and output)
   int block_length;//N; //!< Length of input/output sequence
   int block_length_w_tail;//block_length with tailing
   int k; //input
   int n; //output
   int m; //m highest degree polynomial of G(D)
   int no_states;
   int recv_sequence; //length of received sequence
   int encoding_steps; //number of steps in the trellis diagram for the data not the tailing off. The number of hops from the trellis diagram
   std::string ff_octal;
   std::string fb_octal;
   libbase::matrix<std::string> ffcodebook; //Feedforward connection string
   libbase::matrix<std::string> fbcodebook; //Feedback connection string
   libbase::matrix<bool> statetable;
   /*Conv Codes parameters - END*/

   /*Conv Codes Functions - BEGIN*/
   void encode_data(const array1i_t& source, array1i_t& encoded);

   void feedforward(std::istream& sin);
   void feedback(std::istream& sin);

   void init_matrices(libbase::matrix<std::vector<dbl> >& gamma, libbase::matrix<dbl>& alpha, libbase::matrix<dbl>& beta, libbase::matrix<dbl>& output_symbol, libbase::matrix<dbl>& output_bit);
   void init_gamma(libbase::matrix<std::vector<dbl> >& gamma, dbl value);
   void init_alpha(libbase::matrix<dbl>& alpha);
   void init_beta(libbase::matrix<dbl>& beta);
   void init_output_symbol(libbase::matrix<dbl>& output_symbol);
   void init_output_bit(libbase::matrix<dbl>& output_bit);

   void work_gamma(libbase::matrix<std::vector<dbl> >& gamma, array1vd_t& recv_ptable);
   dbl calc_gamma_prob(int state_table_row, int col, array1vd_t& recv_ptable);
   double calc_gamma_AWGN(int state_table_row, int col, double* recv, double Lc);
   void work_alpha(libbase::matrix<std::vector<dbl> >& gamma, libbase::matrix<dbl>& alpha);
   void work_beta(libbase::matrix<std::vector<dbl> >& gamma, libbase::matrix<dbl>& beta);

   void decode(libbase::matrix<std::vector<dbl> >& gamma, libbase::matrix<dbl>& alpha, libbase::matrix<dbl>& beta, libbase::matrix<dbl>& output_symbol);

   void decode_normalise(libbase::matrix<std::vector<dbl> >& gamma, libbase::matrix<dbl>& alpha, libbase::matrix<dbl>& beta, libbase::matrix<dbl>& output_symbol);

   void decode(libbase::matrix<std::vector<dbl> >& gamma, libbase::matrix<dbl>& alpha, libbase::matrix<dbl>& beta, libbase::matrix<dbl>& output_symbol, array1vd_t& output_posteriors);
   void multiple_inputs(libbase::matrix<dbl>& output_symbol, libbase::matrix<dbl>& output_bit);
   void fill_ptable(array1vd_t& ptable, libbase::matrix<dbl>& output_bit);
   void work_softout(libbase::matrix<dbl>& output_bit, libbase::vector<dbl>& softout);

   void normalize(libbase::matrix<dbl>& mat);
   int get_next_state(int input, int curr_state);
   int get_next_state(int input, int curr_state, int& state_table_row);
   std::string get_output(int input, int curr_state);

   std::string oct2bin(std::string input, int size, int type);
   int bin2int(std::string binary);
   std::string int2bin(int input, int size);
   bool toBool(char const& bit);
   void fill_state_diagram_fb();
   void fill_state_diagram_ff(int* m_arr);
   void disp_statetable();
   std::string toString(int number);
   char toChar(bool bit);
   int toInt(bool bit);
   void writetofile(array1vd_t& received, array1vd_t& decoded);
   void datatofile(const array1i_t& source, array1i_t& encoded);
   //void settoval(libbase::matrix<double>& mat, double value);
   /*Conv Codes Functions - END*/

protected:
   // Internal codec_softout operations
   void resetpriors();
   void setpriors(const array1vd_t& ptable);
   void setreceiver(const array1vd_t& ptable);
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
   void do_init_decoder(const array1vd_t& ptable)
      {
      setreceiver(ptable);
      resetpriors();
      }
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app)
      {
      setreceiver(ptable);
      setpriors(app);
      }
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   conv_codec(int q=2, int N=1) :
      alphabet_size(q), block_length(N)
      {
      }
   // @}

   // Codec operations
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> input_block_size() const
      {
      return libbase::size_type<libbase::vector>(block_length);
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(block_length_w_tail);
      }
   int num_inputs() const
      {
      return alphabet_size;
      }
   int num_outputs() const
      {
      return alphabet_size;
      }
   int tail_length() const
      {
      return 0;
      }
   int num_iter() const
      {
      return 1;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(conv_codec)
};

} // end namespace

#endif

