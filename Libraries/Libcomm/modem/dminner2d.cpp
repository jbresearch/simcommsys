/*!
 * \file
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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "dminner2d.h"
#include "channel/bsid2d.h"
#include "dminner2.h"
#include "timer.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show marker & received rows/cols as they are being decoded
// 3 - Also show input/output probability tables
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \copydoc blockmodem::advance()
 * 
 * Updates 2D marker sequence
 */

template <class real>
void dminner2d<real>::advance() const
   {
   // Inherit sizes
   const int N = this->input_block_size().cols();
   const int M = this->input_block_size().rows();
   // Initialize space
   marker.init(M * m, N * n);
   // creates 'tau' elements of 'n' bits each
   for (int j = 0; j < marker.size().rows(); j++)
      for (int i = 0; i < marker.size().cols(); i++)
         marker(j, i) = (r.ival(2) != 0);
   }

/*!
 * \copydoc blockmodem::domodulate()
 * 
 * Performs modulation; this is similar to the original (1D) DM code, except
 * that here both the input sequence, marker sequence, and codewords are
 * two-dimensional:
 * - input sequence (from codec) is M x N symbols
 * - codewords are each m x n bits
 * - marker and output (transmitted) sequences are therefore M m  x N n bits
 */

template <class real>
void dminner2d<real>::domodulate(const int q,
      const libbase::matrix<int>& encoded, libbase::matrix<bool>& tx)
   {
   // Each 'encoded' symbol must be representable by a single codeword
   assertalways(this->q == q);
   // Inherit sizes
   const int N = this->input_block_size().cols();
   const int M = this->input_block_size().rows();
   // Check validity
   assertalways(N == encoded.size().cols());
   assertalways(M == encoded.size().rows());
   // Initialise result vector
   tx = marker;
   assert(tx.size().cols() == N * n);
   assert(tx.size().rows() == M * m);
   // Encode source
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         for (int ii = 0; ii < m; ii++)
            for (int jj = 0; jj < n; jj++)
               tx(i * m + ii, j * n + jj) ^= codebook(encoded(i, j))(ii, jj);
   }

/*!
 * \copydoc blockmodem::dodemodulate()
 * 
 * Wrapper for decoder function assuming equiprobable APP table.
 */

template <class real>
void dminner2d<real>::dodemodulate(const channel<bool, libbase::matrix>& chan,
      const libbase::matrix<bool>& rx, libbase::matrix<array1d_t>& ptable)
   {
   // Inherit sizes
   const int N = this->input_block_size().cols();
   const int M = this->input_block_size().rows();
   // Create equiprobable a-priori probability table
   libbase::matrix<array1d_t> app(M, N);
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         {
         app(i, j).init(q);
         app(i, j) = 1.0;
         }
   // Now call the full decode function
   dodemodulate(chan, rx, app, ptable);
   }

/*!
 * \copydoc blockmodem::dodemodulate()
 * 
 * Decodes 2D sequence by performing iterative row/column decodings, given the
 * a-priori probabilities for each possible value of encoded symbols in app.
 * Output probabilities for each possible transmitted symbols is returned
 * in ptable. Decoding is performed as follows:
 * 
 * 1) Initialize output (ptable) from the input (app) probability table
 * 2) Consider each encoded row of symbols, one at a time (call this 'i'):
 * a) Extract the corresponding row of probabilities from ptable, to
 * use with the row decoder (call this 'pacc')
 * b) Consider each transmitted row of bits for this row of symbols,
 * one at a time (call this 'ii'):
 * i) Construct a 1D DM decoder for the row, where the codebook
 * is extracted from the corresponding row of the 2D codebook
 * and the marker sequence is extracted from the
 * corresponding row of the 2D marker sequence.
 * ii) Decode using the row decoder
 * iii) Update 'pacc' by multiplying with the corresponding output
 * from the row decoder
 * c) Update ptable by copying 'pacc' back in place
 * 3) Consider each encoded column in the same way as with step 2
 * 4) Repeat from step 2 for a given number of iterations
 */

template <class real>
void dminner2d<real>::dodemodulate(const channel<bool, libbase::matrix>& chan,
      const libbase::matrix<bool>& rx, const libbase::matrix<array1d_t>& app,
      libbase::matrix<array1d_t>& ptable)
   {
   // Inherit sizes
   const int N = this->input_block_size().cols();
   const int M = this->input_block_size().rows();
   // Check input validity
   assertalways(N == app.size().cols());
   assertalways(M == app.size().rows());
   // Copy channel and create a 1D one with same parameters
   bsid2d theirchan = dynamic_cast<const bsid2d&> (chan);
   bsid mychan;
   mychan.set_ps(theirchan.get_ps());
   mychan.set_pd(theirchan.get_pd());
   mychan.set_pi(theirchan.get_pi());
   // Initialize result vector
   ptable = app;
   // Temporary variables
   libbase::vector<bool> rxvec;
   libbase::vector<bool> markervec;
   libbase::vector<array1d_t> pin;
   libbase::vector<array1d_t> pout;
   libbase::vector<array1d_t> pacc;
   dminner2<real> rowdec(n, int(log2(q)));
   dminner2<real> coldec(m, int(log2(q)));
   rowdec.set_thresholds(0, 0);
   coldec.set_thresholds(0, 0);
   // Iterate as requested
   for (int k = 0; k < this->iter; k++)
      {
      // Decode rows
      mychan.set_blocksize(M);
      for (int i = 0; i < M; i++)
         {
         ptable.extractrow(pin, i);
         // initialize storage
         libbase::allocate(pacc, pin.size(), pin(0).size());
         // initialize value
         pacc = 1;
         for (int ii = 0; ii < m; ii++)
            {
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): Decoding row " << ii << " of symbol row " << i << std::endl;
#endif
            // set up row decoder's codebook & marker sequence
            rowdec.set_codebook(get_alphabet_row(ii));
            marker.extractrow(markervec, i * m + ii);
            rowdec.set_marker(markervec);
            rx.extractrow(rxvec, i * m + ii);
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): marker = " << markervec;
            libbase::trace << "DEBUG (dminner2d): rx = " << rxvec;
#endif
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pin = " << pin;
#endif
            rowdec.demodulate(mychan, rxvec, pin, pout);
            pacc *= pout;
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pout = " << pout;
            libbase::trace << "DEBUG (dminner2d): pacc = " << pacc;
#endif
            }
         ptable.insertrow(pacc, i);
         }
      // Decode columns
      mychan.set_blocksize(N);
      for (int j = 0; j < N; j++)
         {
         ptable.extractcol(pin, j);
         // initialize storage
         libbase::allocate(pacc, pin.size(), pin(0).size());
         // initialize value
         pacc = 1;
         for (int jj = 0; jj < n; jj++)
            {
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): Decoding col " << jj << " of symbol col " << j << std::endl;
#endif
            // set up col decoder's codebook & marker sequence
            coldec.set_codebook(get_alphabet_col(jj));
            marker.extractcol(markervec, j * n + jj);
            coldec.set_marker(markervec);
            rx.extractcol(rxvec, j * n + jj);
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): marker = " << markervec;
            libbase::trace << "DEBUG (dminner2d): rx = " << rxvec;
#endif
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pin = " << pin;
#endif
            coldec.demodulate(mychan, rxvec, pin, pout);
            pacc *= pout;
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pout = " << pout;
            libbase::trace << "DEBUG (dminner2d): pacc = " << pacc;
#endif
            }
         ptable.insertcol(pacc, j);
         }
      }
   }

/*!
 * \brief Confirm that codebook is valid
 * Checks that all codebook entries are of the correct size and that there are no
 * duplicate entries.
 */

template <class real>
void dminner2d<real>::validatecodebook() const
   {
   assertalways(codebook.size() == num_symbols());
   for (int i = 0; i < codebook.size(); i++)
      {
      // all entries should be of the correct size
      assertalways(codebook(i).size().cols() == n);
      assertalways(codebook(i).size().rows() == m);
      // all entries should be distinct
      for (int j = 0; j < i; j++)
         assertalways(codebook(i).isnotequalto(codebook(j)));
      }
   }

template <class real>
libbase::vector<libbase::bitfield> dminner2d<real>::get_alphabet_row(int i) const
   {
   libbase::vector<libbase::bitfield> codebook_b(codebook.size());
   for (int k = 0; k < codebook.size(); k++)
      codebook_b(k) = libbase::bitfield(codebook(k).extractrow(i));
   return codebook_b;
   }

template <class real>
libbase::vector<libbase::bitfield> dminner2d<real>::get_alphabet_col(int j) const
   {
   libbase::vector<libbase::bitfield> codebook_b(codebook.size());
   for (int k = 0; k < codebook.size(); k++)
      codebook_b(k) = libbase::bitfield(codebook(k).extractcol(j));
   return codebook_b;
   }

/*!
 * \brief Object initialization
 * Determines code parameters from codebook and sets up object for use.
 * This includes validating the codebook, setting up the marker generator,
 * and clearing the marker sequence.
 */

template <class real>
void dminner2d<real>::init()
   {
   // Determine code parameters from codebook
   q = codebook.size();
   assertalways(q > 0);
   m = codebook(0).size().rows();
   n = codebook(0).size().cols();
   // Validate codebook
   validatecodebook();
   // Seed the marker generator and clear the sequence
   r.seed(0);
   marker.init(0, 0);
   }

// description output

template <class real>
std::string dminner2d<real>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative 2D DM Inner Code (";
   sout << m << "x" << n << "/" << q << ", ";
   sout << codebookname << " codebook)";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& dminner2d<real>::serialize(std::ostream& sout) const
   {
   sout << "# Number of iterations" << std::endl;
   sout << iter << std::endl;
   sout << "# Name of the code book" << std::endl;
   sout << codebookname << std::endl;
   sout << "# The code book entries" << std::endl;
   sout << codebook << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& dminner2d<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> iter >> libbase::verify;
   sin >> libbase::eatcomments >> codebookname >> libbase::verify;
   sin >> libbase::eatcomments >> codebook >> libbase::verify;
   init();
   return sin;
   }

} // end namespace

#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#ifdef USE_CUDA
#define REAL_TYPE_SEQ \
   (float)(double)
#else
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)
#endif

/* Serialization string: dminner2d<real,norm>
 * where:
 *      real = float | double | logrealfast (CPU only)
 */
#define INSTANTIATE(r, x, type) \
      template class dminner2d<type>; \
      template <> \
      const serializer dminner2d<type>::shelper( \
            "blockmodem", \
            "dminner2d<" BOOST_PP_STRINGIZE(type) ">", \
            dminner2d<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
