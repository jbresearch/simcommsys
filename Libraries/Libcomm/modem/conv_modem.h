/*!
 * \file
 * $Id: tvb.h 9935 2013-09-26 13:56:31Z jabriffa $
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

#ifndef __conv_modem_h
#define __conv_modem_h

#include "config.h"

#include "stream_modulator.h"
#include "channel/qids.h"
#include "algorithm/fba2-interface.h"

#include "randgen.h"
#include "itfunc.h"
#include "vector_itfunc.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>
#include <memory>

#include "boost/shared_ptr.hpp"

//#ifdef USE_CUDA
//#  include "tvb-receiver-cuda.h"
//#else
#  include "tvb-receiver.h"
//#endif

#include "conv_modem_class.h"

#define swap_ls(a,b) swap_temp = (a);      \
   (a) = (b);                            \
   (b) = swap_temp;


namespace libcomm {

/*!
 * \brief   Time-Varying Block Code.
 * \author  Johann Briffa
 * $Id: tvb.h 9935 2013-09-26 13:56:31Z jabriffa $
 *
 * Implements a MAP decoding algorithm for a generalized class of
 * synchronization-correcting codes. The algorithm is described in
 * Briffa et al, "A MAP Decoder for a General Class of Synchronization-
 * Correcting Codes", Submitted to Trans. IT, 2011.
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 */

template <class sig, class real, class real2>
class conv_modem : public stream_modulator<sig> , public parametric {
private:
   // Shorthand for class hierarchy
   typedef stream_modulator<sig> Interface;
   typedef conv_modem<sig, real, real2> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<array1s_t> array1vs_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   enum codebook_t {
      codebook_sparse = 0, //!< as in DM construction
      codebook_random, //!< randomly-constructed codebooks, update every frame
      codebook_user_sequential, //!< sequentially-applied user codebooks
      codebook_user_random, //!< randomly-applied user codebooks
      codebook_undefined
   };
   enum marker_t {
      marker_zero = 0, //!< no marker sequence
      marker_random, //!< random marker sequence, update every frame
      marker_user_sequential, //!< sequentially-applied user sequence
      marker_user_random, //!< randomly-applied user sequence
      marker_undefined
   };
   enum storage_t {
      storage_local = 0, //!< always use local storage
      storage_global, //!< always use global storage
      storage_conditional, //!< use global storage below memory limit
      storage_undefined
   };
   // @}
private:
   /*! \name User-defined parameters */
   //int n; //!< codeword length in symbols
   //int q; //!< number of codewords (input alphabet size)
   real th_inner; //!< Threshold factor for inner cycle
   real th_outer; //!< Threshold factor for outer cycle
   double Pr; //!< Probability of channel event outside chosen limits
   int lookahead; //!< Number of codewords to look ahead when stream decoding
   mutable libbase::randgen r; //!< for construction and random application of codebooks and marker sequence

private:
   // Atomic modem operations (private as these should never be used)
   const sig modulate(const int index) const
      {
      failwith("Function should not be used.");
      return sig();
      }
   const int demodulate(const sig& signal) const
      {
      failwith("Function should not be used.");
      return 0;
      }
   const int demodulate(const sig& signal, const array1d_t& app) const
      {
      failwith("Function should not be used.");
      return 0;
      }
protected:
   // Interface with derived classes
   void advance() const;
   void domodulate(const int N, const array1i_t& encoded, array1s_t& tx);
   void dodemodulate(const channel<sig>& chan, const array1s_t& rx,
         array1vd_t& ptable);
   void dodemodulate(const channel<sig>& chan, const array1s_t& rx,
         const array1vd_t& app, array1vd_t& ptable);
   void dodemodulate(const channel<sig>& chan, const array1s_t& rx,
         const libbase::size_type<libbase::vector> lookahead,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
         array1d_t& eof_post, const libbase::size_type<libbase::vector> offset);

public:
   /*! \name Marker-specific setup functions */
   void set_thresholds(const real th_inner, const real th_outer);
   void set_parameter(const double x)
      {
      set_thresholds(real(x), real(x));
      }
   double get_parameter() const
      {
      assert(th_inner == th_outer);
      return th_inner;
      }
   // @}

   // Setup functions
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   // Informative functions
   int num_symbols() const
      {
      return 2;
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(this->input_block_size() * n);
      }
   double energy() const
      {
      return n;
      }

   // Block modem operations - streaming extensions
   void get_post_drift_pdf(array1vd_t& pdftable) const
      {
      //// Inherit sizes
      //const int N = this->input_block_size();
      //// get the posterior channel drift pdf at codeword boundaries
      //array1vr_t pdftable_r;
      //fba_ptr->get_drift_pdf(pdftable_r);
      //libbase::normalize_results(pdftable_r.extract(0, N + 1), pdftable);
      }
   array1i_t get_boundaries(void) const
      {
      // Inherit sizes
      const int N = this->input_block_size();
      // construct list of codeword boundary positions
      array1i_t postable(N + 1);
      for (int i = 0; i <= N; i++)
         postable(i) = i * n;
      return postable;
      }
   libbase::size_type<libbase::vector> get_suggested_lookahead(void) const
      {
      return libbase::size_type<libbase::vector>(n * lookahead);
      }
   double get_suggested_exclusion(void) const
      {
      return Pr;
      }

   // Description
   std::string description() const;
public:
   typedef vector<vector<vector<Gamma_Storage> > > vector_3d;
   typedef vector<b_storage> vec_b_storage;
private:
      /*Conv Codes parameters - BEGIN*/
      int type;
      int alphabet_size; //!< Alphabet size (input and output)
      int block_length;//N; //!< Length of input/output sequence
      int block_length_w_tail;//block_length with tailing
      int k; //input
      int n; //output
      int m; //m highest degree polynomial of G(D)
      int no_states;
      int encoding_steps; //number of steps in the trellis diagram for the data not the tailing off. The number of hops from the trellis diagram
      std::string ff_octal;
      std::string fb_octal;
      libbase::matrix<std::string> ffcodebook; //Feedforward connection string
      libbase::matrix<std::string> fbcodebook; //Feedback connection string
      libbase::matrix<bool> statetable;

      vector_3d gamma_storage;

      typename qids<sig, real2>::metric_computer computer;
      qids<sig,real2> mychan;

      unsigned int no_del;//max num del
      unsigned int no_ins;//max num ins
      unsigned int rho;//max allowable symbol shift
      /*Conv Codes parameters - END*/
      
       /*Conv Codes Functions - BEGIN*/
      void feedforward(std::istream& sin);
      
      void encode_data(const array1i_t& encoded, array1s_t& tx);

      double get_gamma(unsigned int cur_state, unsigned int cur_bs, unsigned int next_state, unsigned int next_bs, array1s_t& orig_seq, array1s_t& recv_seq);
      double work_gamma(array1s_t& orig_seq, array1s_t& recv_seq);
      
      int conv_modem<sig, real, real2>::sleven(std::string string1, std::string string2, int sub, int ins, int del);

      int get_next_state(int input, int curr_state);
      
      unsigned int get_input(unsigned int cur_state, unsigned int prev_state);

      void get_output(int input, int curr_state, array1s_t& output);
      void get_received(unsigned int b, unsigned int cur_bs, unsigned int next_bs, unsigned int no_del, const array1s_t& rx, array1s_t& recv_codeword);//Get the bits from the received sequence considering ins/dels
      std::string oct2bin(std::string input, int size, int type);
      int bin2int(std::string binary);
      std::string int2bin(int input, int size);
      bool toBool(char const& bit);
      void fill_state_diagram_ff(int* m_arr);
      void disp_statetable();
      std::string toString(int number);
      char toChar(bool bit);
      int toInt(bool bit);

      void print_sig(array1s_t& data);
      /*Conv Codes Functions - END*/

//Serialization Support
DECLARE_SERIALIZER(conv_modem)
};

} // end namespace

#endif
