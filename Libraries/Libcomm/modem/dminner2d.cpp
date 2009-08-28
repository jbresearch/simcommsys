/*!
 * \file
 * 
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "dminner2d.h"
#include "channel/bsid2d.h"
#include "dminner2.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show pilot & received rows/cols as they are being decoded
// 3 - Also show input/output probability tables
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \copydoc blockmodem::advance()
 * 
 * Updates 2D pilot sequence
 */

template <class real, bool norm>
void dminner2d<real, norm>::advance() const
   {
   // Inherit sizes
   const int M = this->input_block_size().cols();
   const int N = this->input_block_size().rows();
   // Initialize space
   pilot.init(N * n, M * m);
   // creates 'tau' elements of 'n' bits each
   for (int i = 0; i < pilot.size().cols(); i++)
      for (int j = 0; j < pilot.size().rows(); j++)
         pilot(j, i) = (r.ival(2) != 0);
   }

/*!
 * \copydoc blockmodem::domodulate()
 * 
 * Performs modulation; this is similar to the original (1D) DM code, except
 * that here both the input sequence, pilot sequence, and sparse symbols are
 * two-dimensional:
 * - input sequence (from codec) is M x N symbols
 * - sparse symbols are each m x n bits
 * - pilot and output (transmitted) sequences are therefore M m  x N n bits
 */

template <class real, bool norm>
void dminner2d<real, norm>::domodulate(const int q,
      const libbase::matrix<int>& encoded, libbase::matrix<bool>& tx)
   {
   // Each 'encoded' symbol must be representable by a single sparse matrix
   assertalways(this->q == q);
   // Inherit sizes
   const int M = this->input_block_size().cols();
   const int N = this->input_block_size().rows();
   // Check validity
   assertalways(M == encoded.size().cols());
   assertalways(N == encoded.size().rows());
   // Initialise result vector
   tx = pilot;
   assert(tx.size().cols() == M*m);
   assert(tx.size().rows() == N*n);
   // Encode source
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         for (int ii = 0; ii < m; ii++)
            for (int jj = 0; jj < n; jj++)
               tx(j * n + jj, i * m + ii) ^= lut(encoded(j, i))(jj, ii);
   }

/*!
 * \copydoc blockmodem::dodemodulate()
 * 
 * Wrapper for decoder function assuming equiprobable APP table.
 */

template <class real, bool norm>
void dminner2d<real, norm>::dodemodulate(
      const channel<bool, libbase::matrix>& chan,
      const libbase::matrix<bool>& rx, libbase::matrix<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->input_block_size().cols();
   const int N = this->input_block_size().rows();
   // Create equiprobable a-priori probability table
   libbase::matrix<array1d_t> app(N, M);
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         {
         app(j, i).init(q);
         app(j, i) = 1.0;
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
 * i) Construct a 1D DM decoder for the row, where the sparse symbol
 * alphabet is extracted from the corresponding row of the 2D
 * alphabet, and the pilot sequence is extracted from the
 * corresponding row of the 2D pilot sequence.
 * ii) Decode using the row decoder
 * iii) Update 'pacc' by multiplying with the corresponding output
 * from the row decoder
 * c) Update ptable by copying 'pacc' back in place
 * 3) Consider each encoded column in the same way as with step 2
 * 4) Repeat from step 2 for a given number of iterations
 */

template <class real, bool norm>
void dminner2d<real, norm>::dodemodulate(
      const channel<bool, libbase::matrix>& chan,
      const libbase::matrix<bool>& rx, const libbase::matrix<array1d_t>& app,
      libbase::matrix<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->input_block_size().cols();
   const int N = this->input_block_size().rows();
   // Check input validity
   assertalways(M == app.size().cols());
   assertalways(N == app.size().rows());
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
   libbase::vector<bool> wsvec;
   libbase::vector<array1d_t> pin;
   libbase::vector<array1d_t> pout;
   libbase::vector<array1d_t> pacc;
   dminner2<real, norm> rowdec(n, int(log2(q)));
   dminner2<real, norm> coldec(m, int(log2(q)));
   rowdec.set_thresholds(0, 0);
   coldec.set_thresholds(0, 0);
   // Iterate as requested
   const int iter = 1;
   for (int k = 0; k < iter; k++)
      {
      // Decode rows
      mychan.set_blocksize(N);
      for (int i = 0; i < M; i++)
         {
         ptable.extractrow(pin, i);
         // initialize storage
         pacc.init(pin.size());
         for (int ii = 0; ii < pacc.size(); ii++)
            pacc(ii).init(pin(ii).size());
         // initialize value
         pacc = 1;
         for (int ii = 0; ii < m; ii++)
            {
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): Decoding row " << ii << " of symbol row " << i << "\n";
#endif
            // set up row decoder's sparse alphabet & pilot sequence
            rowdec.set_lut(get_alphabet_row(ii));
            pilot.extractrow(wsvec, i * m + ii);
            rowdec.set_pilot(wsvec);
            rx.extractrow(rxvec, i * m + ii);
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): pilot = " << wsvec;
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
      mychan.set_blocksize(M);
      for (int j = 0; j < N; j++)
         {
         ptable.extractcol(pin, j);
         // initialize storage
         pacc.init(pin.size());
         for (int jj = 0; jj < pacc.size(); jj++)
            pacc(jj).init(pin(jj).size());
         // initialize value
         pacc = 1;
         for (int jj = 0; jj < n; jj++)
            {
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): Decoding col " << jj << " of symbol col " << j << "\n";
#endif
            // set up col decoder's sparse alphabet & pilot sequence
            coldec.set_lut(get_alphabet_col(jj));
            pilot.extractcol(wsvec, j * n + jj);
            coldec.set_pilot(wsvec);
            rx.extractcol(rxvec, j * n + jj);
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): pilot = " << wsvec;
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
 * \brief Confirm that LUT is valid
 * Checks that all LUT entries are of the correct size and that there are no
 * duplicate entries.
 */

template <class real, bool norm>
void dminner2d<real, norm>::validatelut() const
   {
   assertalways(lut.size() == num_symbols());
   for (int i = 0; i < lut.size(); i++)
      {
      // all entries should be of the correct size
      assertalways(lut(i).size().cols() == m);
      assertalways(lut(i).size().rows() == n);
      // all entries should be distinct
      for (int j = 0; j < i; j++)
         assertalways(lut(i).isnotequalto(lut(j)));
      }
   }

template <class real, bool norm>
libbase::vector<libbase::bitfield> dminner2d<real, norm>::get_alphabet_row(
      int i) const
   {
   libbase::vector<libbase::bitfield> lutb(lut.size());
   for (int k = 0; k < lut.size(); k++)
      lutb(k) = libbase::bitfield(lut(k).extractrow(i));
   return lutb;
   }

template <class real, bool norm>
libbase::vector<libbase::bitfield> dminner2d<real, norm>::get_alphabet_col(
      int j) const
   {
   libbase::vector<libbase::bitfield> lutb(lut.size());
   for (int k = 0; k < lut.size(); k++)
      lutb(k) = libbase::bitfield(lut(k).extractcol(j));
   return lutb;
   }

/*!
 * \brief Object initialization
 * Determines code parameters from LUT and sets up object for use.
 * This includes validating the LUT, setting up the pilot generator,
 * and clearing the pilot sequence.
 */

template <class real, bool norm>
void dminner2d<real, norm>::init()
   {
   // Determine code parameters from LUT
   q = lut.size();
   assertalways(q > 0);
   m = lut(0).size().cols();
   n = lut(0).size().rows();
   // Validate LUT
   validatelut();
   // Seed the pilot generator and clear the sequence
   r.seed(0);
   pilot.init(0, 0);
   }

// description output

template <class real, bool norm>
std::string dminner2d<real, norm>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative 2D DM Inner Code (";
   sout << m << "x" << n << "/" << q << ", ";
   sout << lutname << " codebook)";
   return sout.str();
   }

// object serialization - saving

template <class real, bool norm>
std::ostream& dminner2d<real, norm>::serialize(std::ostream& sout) const
   {
   sout << lutname;
   sout << lut;
   return sout;
   }

// object serialization - loading

template <class real, bool norm>
std::istream& dminner2d<real, norm>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> lutname;
   sin >> libbase::eatcomments >> lut;
   init();
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2d<logrealfast, false> ;
template <>
const serializer dminner2d<logrealfast, false>::shelper = serializer(
      "blockmodem", "dminner2d<logrealfast>",
      dminner2d<logrealfast, false>::create);

template class dminner2d<double, true> ;
template <>
const serializer dminner2d<double, true>::shelper = serializer("blockmodem",
      "dminner2d<double>", dminner2d<double, true>::create);

} // end namespace
