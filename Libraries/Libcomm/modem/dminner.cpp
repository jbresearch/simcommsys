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

#include "dminner.h"
#include "sparse.h"
#include "timer.h"
#include "pacifier.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show codebooks on manual update
// 3 - Show input and output of encoding process
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// internally-used functions

/*!
 * \brief Return encoding of data 'd' at position 'i'
 */
template <class real>
libbase::vector<bool> dminner<real>::encode(const int i, const int d) const
   {
#ifndef NDEBUG
   // Inherit sizes
   const int q = 1 << k;
   const int tau = this->input_block_size();
   // validate input
   assert(i >= 0 && i < tau);
   assert(d >= 0 && d < q);
#endif
   // extract marker sequence and codeword
   const int w = marker(i);
   const int s = codebook(i % num_codebooks(), d);
#if DEBUG>=3
   libbase::trace << "DEBUG (dminner::encode): word " << i << "\t";
   libbase::trace << "s = " << libbase::bitfield(s, n) << "\t";
   libbase::trace << "w = " << libbase::bitfield(w, n) << std::endl;
#endif
   // 'tx' is the vector of transmitted symbols that we're considering
   libbase::bitfield tx(w ^ s, n);
   // NOTE: we transmit the low-order bits first
   return tx.asvector();
   }

/*!
 * \brief Check that all entries in table have correct length
 */
template <class real>
void dminner<real>::validate_bitfield_length(const libbase::vector<
      libbase::bitfield>& table) const
   {
   assertalways(table.size() > 0);
   for (int i = 0; i < table.size(); i++)
      assertalways(table(i).size() == n);
   }

/*!
 * \brief Set up marker sequence for the current frame as given
 */
template <class real>
void dminner<real>::copymarker(
      const libbase::vector<libbase::bitfield>& marker_b)
   {
   validate_bitfield_length(marker_b);
   marker = marker_b;
   }

/*!
 * \brief Set up codebook with the given codewords
 */
template <class real>
void dminner<real>::copycodebook(const int i, const libbase::vector<
      libbase::bitfield>& codebook_b)
   {
   assertalways(codebook_b.size() == num_symbols());
   validate_bitfield_length(codebook_b);
   // convert to vector of integers
   const libbase::vector<int> codebook_i(codebook_b);
   // insert into matrix
   assertalways(i >= 0 && i < num_codebooks());
   assertalways(codebook.size().cols() == num_symbols());
   codebook.insertrow(codebook_i, i);
   }

/*!
 * \brief Display codebook on given stream
 */

template <class real>
void dminner<real>::showcodebook(std::ostream& sout) const
   {
   assert(num_codebooks() >= 1);
   assert(codebook.size().cols() == num_symbols());
   for (int i = 0; i < num_codebooks(); i++)
      {
      sout << "codebook " << i << " (k=" << k << ", n=" << n << "):"
            << std::endl;
      for (int d = 0; d < num_symbols(); d++)
         sout << d << "\t" << libbase::bitfield(codebook(i, d), n) << "\t"
               << libbase::weight(codebook(i, d)) << std::endl;
      }
   }

/*!
 * \brief Confirm that codebook is valid
 * Checks that all codebook entries are within range and that there are no
 * duplicate entries.
 */

template <class real>
void dminner<real>::validatecodebook() const
   {
   assertalways(num_codebooks() >= 1);
   assertalways(codebook.size().cols() == num_symbols());
   for (int i = 0; i < num_codebooks(); i++)
      for (int d = 0; d < num_symbols(); d++)
         {
         // all entries should be within size
         assertalways(codebook(i,d) >= 0 && codebook(i,d) < (1<<n));
         // all entries should be distinct for each codebook index
         for (int dd = 0; dd < d; dd++)
            assertalways(codebook(i, dd) != codebook(i, d));
         }
   }

//! Compute and update mean density of codebook

template <class real>
void dminner<real>::computemeandensity()
   {
   array2i_t w = codebook;
   w.apply(libbase::weight);
   f = w.sum() / double(n * w.size());
#ifndef NDEBUG
   if (n > 2)
      libbase::trace << "Watermark code density = " << f << std::endl;
#endif
   }

//! Inform user if I or xmax have changed

template <class real>
void dminner<real>::checkforchanges(int I, int xmax) const
   {
   static int last_I = 0;
   static int last_xmax = 0;
   if (last_I != I || last_xmax != xmax)
      {
      std::cerr << "DMinner: I = " << I << ", xmax = " << xmax << std::endl;
      last_I = I;
      last_xmax = xmax;
      }
   }

template <class real>
void dminner<real>::work_results(const array1b_t& r, array1vr_t& ptable,
      const int xmax, const int dxmax, const int I) const
   {
   libbase::pacifier progress("FBA Results");
   // local flag for path thresholding
   const bool thresholding = (th_outer > real(0));
   // determine limits
   const int dmin = std::max(-n, -dxmax);
   const int dmax = std::min(n * I, dxmax);
   // Inherit block size from last modulation step
   const int q = 1 << k;
   const int N = marker.size();
   // Initialise result vector (one symbol per timestep)
   libbase::allocate(ptable, N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (int x1 = -xmax; x1 <= xmax; x1++)
            if (FBA::getF(n * i, x1) > threshold)
               threshold = FBA::getF(n * i, x1);
         threshold *= th_outer;
         }
      for (int d = 0; d < q; d++)
         {
         real p = 0;
         // create the considered transmitted sequence
         array1b_t t = encode(i, d);
         // event must fit the received sequence:
         // (this is limited to start and end conditions)
         // 1. n*i+x1 >= 0
         // 2. n*(i+1)-1+x2 <= r.size()-1
         // limits on insertions and deletions must be respected:
         // 3. x2-x1 <= n*I
         // 4. x2-x1 >= -n
         // limits on introduced drift in this section:
         // (necessary for forward recursion on extracted segment)
         // 5. x2-x1 <= dxmax
         // 6. x2-x1 >= -dxmax
         const int x1min = std::max(-xmax, -n * i);
         const int x1max = xmax;
         const int x2max_bnd = std::min(xmax, r.size() - n * (i + 1));
         for (int x1 = x1min; x1 <= x1max; x1++)
            {
            const real F = FBA::getF(n * i, x1);
            // ignore paths below a certain threshold
            if (thresholding && F < threshold)
               continue;
            const int x2min = std::max(-xmax, dmin + x1);
            const int x2max = std::min(x2max_bnd, dmax + x1);
            for (int x2 = x2min; x2 <= x2max; x2++)
               {
               // compute the conditional probability
               const real R = real(mychan.receive(t, r.extract(n * i + x1, x2
                     - x1 + n)));
               const real B = FBA::getB(n * (i + 1), x2);
               // include the probability for this particular sequence
               p += F * R * B;
               }
            }
         ptable(i)(d) = p;
         }
      }
   if (N > 0)
      std::cerr << progress.update(N, N);
   }

/*!
 * \brief Normalize probability table
 * 
 * The input probability table is normalized such that the largest value is
 * equal to 1; result is converted to double.
 */
template <class real>
void dminner<real>::normalize_results(const array1vr_t& in, array1vd_t& out) const
   {
   const int N = in.size();
   assert(N > 0);
   const int q = in(0).size();
   // check for numerical underflow
   real scale = 0;
   for (int i = 0; i < N; i++)
      scale = std::max(scale, in(i).max());
   assert(scale != real(0));
   scale = real(1) / scale;
   // allocate result space
   libbase::allocate(out, N, q);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      for (int d = 0; d < q; d++)
         out(i)(d) = in(i)(d) * scale;
   }

// initialization / de-allocation

template <class real>
void dminner<real>::init()
   {
   // Fill default codebook if necessary
   if (codebook_type == codebook_sparse)
      {
      libbase::sparse mycodebook(1 << k, n);
      codebook.init(1, num_symbols());
      codebook.insertrow(array1i_t(mycodebook), 0);
      }
#ifndef NDEBUG
   // Display codebook when debugging
   if (n > 2)
      showcodebook(libbase::trace);
#endif
   // Validate codebook and compute the mean density
   validatecodebook();
   computemeandensity();
   // set default thresholds if necessary
   if (!user_threshold)
      {
      th_inner = real(1e-15);
      th_outer = real(1e-6);
      }
   // Seed the generator and clear the marker sequence
   r.seed(0);
   marker.init(0);
   // Check that everything makes sense
   test_invariant();
   }

// Marker-specific setup functions

/*!
 * \copydoc set_marker()
 * \todo Consider moving this method to the dminner2d class
 */
template <class real>
void dminner<real>::set_marker(libbase::vector<bool> marker)
   {
   assertalways((marker.size() % n) == 0);
   // init space for converted vector
   libbase::vector<libbase::bitfield> marker_b(marker.size() / n);
   // convert marker sequence
   for (int i = 0; i < marker_b.size(); i++)
      marker_b(i) = libbase::bitfield(marker.extract(i * n, n));
   // pass through the standard method for setting marker sequence
   set_marker(marker_b);
   }

/*!
 * \brief Overrides the internally-generated marker sequence with given one
 * 
 * The intent of this method is to allow users to apply the dminner decoder
 * in derived algorithms, such as the 2D extension.
 * 
 * \todo merge with copymarker()
 */
template <class real>
void dminner<real>::set_marker(libbase::vector<libbase::bitfield> marker)
   {
   copymarker(marker);
   }

/*!
 * \brief Overrides the codebook with given one
 * 
 * The intent of this method is to allow users to apply the dminner decoder
 * in derived algorithms, such as the 2D extension.
 */
template <class real>
void dminner<real>::set_codebook(libbase::vector<libbase::bitfield> codebook_b)
   {
   // allocate memory and copy read codebook
   codebook.init(1, num_symbols());
   copycodebook(0, codebook_b);
   // update codebook-dependent values
   computemeandensity();
#if DEBUG>=2
   showcodebook(libbase::trace);
#endif
   }

template <class real>
void dminner<real>::set_thresholds(const real th_inner, const real th_outer)
   {
   user_threshold = true;
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   test_invariant();
   }

// implementations of channel-specific metrics for fba

template <class real>
real dminner<real>::R(const int i, const array1b_t& r)
   {
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   const int word = i / n;
   const int bit = i % n;
   bool t = ((marker(word) >> bit) & 1);
   // compute the conditional probability
   return real(mychan.receive(t, r));
   }

// block advance operation - update marker sequence

template <class real>
void dminner<real>::advance() const
   {
   // Inherit sizes
   const int tau = this->input_block_size();
   // Advance marker sequence only for non-zero block sizes
   if (tau > 0)
      {
      // Initialize space
      marker.init(tau);
      switch (marker_type)
         {
         case marker_random:
            // creates 'tau' elements of 'n' bits each
            for (int i = 0; i < tau; i++)
               marker(i) = r.ival(1 << n);
            break;

         case marker_zero:
            marker = 0;
            break;

         case marker_alt_symbol:
            // alternating all-ones and all-zeros for successive symbols
            for (int i = 0; i < tau; i++)
               marker(i) = (i % 2) == 0 ? 0 : (1 << n) - 1;
            break;

         case marker_mod_vec:
            // repeated modification vectors from list
            for (int i = 0; i < tau; i++)
               marker(i) = marker_vectors(i % marker_vectors.size());
            break;

         default:
            failwith("Unknown marker sequence type");
            break;
         }
      }
   }

// encoding and decoding functions

template <class real>
void dminner<real>::domodulate(const int N, const array1i_t& encoded,
      array1b_t& tx)
   {
   // TODO: when N is removed from the interface, rename 'tau' to 'N'
   // Inherit sizes
   const int q = 1 << k;
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == encoded.size());
   // Each 'encoded' symbol must be representable by a single codeword
   assertalways(N == q);
   // Initialise result vector
   tx.init(n * tau);
   assertalways(marker.size() == tau);
   // Encode source stream
   for (int i = 0; i < tau; i++)
      tx.segment(i * n, n) = encode(i, encoded(i));
   }

template <class real>
void dminner<real>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, array1vd_t& ptable)
   {
   // Inherit sizes
   const int N = this->input_block_size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   mychan = dynamic_cast<const bsid&> (chan);
   // Update substitution probability to take into account codeword addition
   const double Ps = mychan.get_ps();
   mychan.set_ps(Ps * (1 - f) + (1 - Ps) * f);
   // Set block size for main forward-backward pass
   mychan.set_blocksize(tau);
   // Determine required FBA parameter values
   const int I = mychan.compute_I(tau);
   const int xmax = mychan.compute_xmax(tau);
   const int dxmax = mychan.compute_xmax(n);
   checkforchanges(I, xmax);
   // Initialize & perform forward-backward algorithm
   FBA::init(tau, I, xmax, th_inner, norm);
   FBA::prepare(rx);
   // Reset substitution probability to original value
   mychan.set_ps(Ps);
   // Set block size for results-computation pass to q-ary symbol size
   mychan.set_blocksize(n);
   // Compute and normalize results
   array1vr_t p;
   work_results(rx, p, xmax, dxmax, I);
   normalize_results(p, ptable);
   }

template <class real>
void dminner<real>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   array1vd_t p;
   // Apply standard demodulation
   This::dodemodulate(chan, rx, p);
   // If we have no prior information, copy results over
   if (app.size() == 0)
      {
      ptable = p;
      return;
      }
   // Multiply-in a-priori probabilities
   const int N = p.size();
   assert(N > 0);
   const int q = p(0).size();
   assert(app.size() == N);
   for (int i = 0; i < N; i++)
      {
      assert(app(i).size() == q);
      for (int d = 0; d < q; d++)
         ptable(i)(d) = p(i)(d) * app(i)(d);
      }
   }

template <class real>
void dminner<real>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const libbase::size_type<
            libbase::vector> offset)
   {
   failwith("Function not implemented.");
   }

// description output

template <class real>
std::string dminner<real>::description() const
   {
   std::ostringstream sout;
   const int q = num_symbols();
   sout << "DM Inner Code (" << n << "," << q << ", ";
   switch (codebook_type)
      {
      case codebook_sparse:
         sout << "sparse codebook";
         break;

      case codebook_user:
         sout << codebookname << " codebook";
         break;

      case codebook_tvb:
         sout << codebookname << " codebook [TVB]";
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   switch (marker_type)
      {
      case marker_random:
         sout << ", random marker";
         break;

      case marker_zero:
         sout << ", no marker";
         break;

      case marker_alt_symbol:
         sout << ", symbol-alternating marker";
         break;

      case marker_mod_vec:
         sout << ", AMVs [" << marker_vectors.size()
               << ", sequential]";
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   if (user_threshold)
      sout << ", thresholds " << th_inner << "/" << th_outer;
   if (norm)
      sout << ", normalized";
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& dminner<real>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 3 << std::endl;
   sout << "# User threshold?" << std::endl;
   sout << user_threshold << std::endl;
   if (user_threshold)
      {
      sout << "#: Inner threshold" << std::endl;
      sout << th_inner << std::endl;
      sout << "#: Outer threshold" << std::endl;
      sout << th_outer << std::endl;
      }
   sout << "# Normalize metrics between time-steps?" << std::endl;
   sout << norm << std::endl;
   sout << "# n" << std::endl;
   sout << n << std::endl;
   sout << "# k" << std::endl;
   sout << k << std::endl;
   sout << "# codebook type (0=sparse, 1=user, 2=tvb)" << std::endl;
   sout << codebook_type << std::endl;
   switch (codebook_type)
      {
      case codebook_sparse:
         break;

      case codebook_user:
         sout << "#: codebook name" << std::endl;
         sout << codebookname << std::endl;
         sout << "#: codebook entries" << std::endl;
         assert(num_codebooks() == 1);
         assert(codebook.size().cols() == num_symbols());
         for (int d = 0; d < num_symbols(); d++)
            sout << libbase::bitfield(codebook(0, d), n) << std::endl;
         break;

      case codebook_tvb:
         sout << "#: codebook name" << std::endl;
         sout << codebookname << std::endl;
         sout << "#: codebook count" << std::endl;
         sout << num_codebooks() << std::endl;
         assert(num_codebooks() >= 1);
         assert(codebook.size().cols() == num_symbols());
         for (int i = 0; i < num_codebooks(); i++)
            {
            sout << "#: codebook entries (table " << i << ")" << std::endl;
            for (int d = 0; d < num_symbols(); d++)
               sout << libbase::bitfield(codebook(i, d), n) << std::endl;
            }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   sout
         << "# marker type (0=random, 1=zero, 2=symbol-alternating, 3=mod-vectors)"
         << std::endl;
   sout << marker_type << std::endl;
   if (marker_type == marker_mod_vec)
      {
      sout << "#: modification vectors" << std::endl;
      sout << marker_vectors.size() << std::endl;
      for (int i = 0; i < marker_vectors.size(); i++)
         sout << libbase::bitfield(marker_vectors(i), n) << std::endl;
      }
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering
 *
 * \version 2 Added marker sequence type
 *
 * \version 3 Added normalization flag
 */

template <class real>
std::istream& dminner<real>::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // handle old-format files (without version number)
   if (version < 2)
      {
      //sin.clear();
      sin.seekg(start);
      version = 1;
      }
   sin >> libbase::eatcomments >> user_threshold >> libbase::verify;
   // deal with inexistent flag as 'false'
   if (sin.fail())
      {
      sin.clear();
      sin.seekg(start);
      user_threshold = false;
      }
   // read or set default thresholds
   if (user_threshold)
      {
      sin >> libbase::eatcomments >> th_inner >> libbase::verify;
      sin >> libbase::eatcomments >> th_outer >> libbase::verify;
      }
   // read decoder parameters
   if (version >= 3)
      sin >> libbase::eatcomments >> norm >> libbase::verify;
   else
      norm = true;
   // read code size
   sin >> libbase::eatcomments >> n >> libbase::verify;
   sin >> libbase::eatcomments >> k >> libbase::verify;
   // read codebook
   int temp;
   sin >> libbase::eatcomments >> temp >> libbase::verify;
   codebook_type = (codebook_t) temp;
   switch (codebook_type)
      {
      case codebook_sparse:
         break;

      case codebook_user:
         {
         sin >> libbase::eatcomments >> codebookname >> libbase::verify;
         // allocate memory
         codebook.init(1, num_symbols());
         // read codebook from stream
         libbase::vector<libbase::bitfield> codebook_b;
         codebook_b.init(num_symbols());
         sin >> libbase::eatcomments;
         codebook_b.serialize(sin);
         libbase::verify(sin);
         // copy read codebook
         copycodebook(0, codebook_b);
         }
         break;

      case codebook_tvb:
         sin >> libbase::eatcomments >> codebookname >> libbase::verify;
         // read codebook count
         sin >> libbase::eatcomments >> temp >> libbase::verify;
         // allocate memory
         codebook.init(temp, num_symbols());
         for (int i = 0; i < num_codebooks(); i++)
            {
            // read codebook from stream
            libbase::vector<libbase::bitfield> codebook_b;
            codebook_b.init(num_symbols());
            sin >> libbase::eatcomments;
            codebook_b.serialize(sin);
            libbase::verify(sin);
            // copy read codebook
            copycodebook(i, codebook_b);
            }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   // read marker sequence type if present
   if (version < 2)
      marker_type = marker_random;
   else
      {
      int temp;
      sin >> libbase::eatcomments >> temp >> libbase::verify;
      marker_type = (marker_t) temp;
      if (marker_type == marker_mod_vec)
         {
         // read modification vectors from stream
         libbase::vector<libbase::bitfield> marker_vectors_b;
         sin >> libbase::eatcomments >> marker_vectors_b >> libbase::verify;
         // copy list of modification vectors
         validate_bitfield_length(marker_vectors_b);
         marker_vectors = marker_vectors_b;
         }
      }
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

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: dminner<real,norm>
 * where:
 *      real = float | double | logrealfast (CPU only)
 */
#define INSTANTIATE(r, x, type) \
      template class dminner<type>; \
      template <> \
      const serializer dminner<type>::shelper( \
            "blockmodem", \
            "dminner<" BOOST_PP_STRINGIZE(type) ">", \
            dminner<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
