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
// 2 - Show LUTs on manual update
// 3 - Show input and output of encoding process
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// internally-used functions

/*!
 * \brief Return encoding of data 'd' at position 'i'
 */
template <class real, bool norm>
libbase::vector<bool> dminner<real, norm>::encode(const int i, const int d) const
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
   const int w = ws(i);
   const int s = lut(i % num_codebooks(), d);
#if DEBUG>=3
   libbase::trace << "DEBUG (dminner::encode): word " << i << "\t";
   libbase::trace << "s = " << libbase::bitfield(s, n) << "\t";
   libbase::trace << "w = " << libbase::bitfield(w, n) << std::endl;
#endif
   // 'tx' is the vector of transmitted symbols that we're considering
   array1b_t tx(n);
   // NOTE: we transmit the low-order bits first
   for (int bit = 0, t = w ^ s; bit < n; bit++, t >>= 1)
      tx(bit) = (t & 1);
   // compute the conditional probability
   return tx;
   }

/*!
 * \brief Check that all entries in table have correct length
 */
template <class real, bool norm>
void dminner<real, norm>::validate_bitfield_length(const libbase::vector<
      libbase::bitfield>& table) const
   {
   assertalways(table.size() > 0);
   for (int i = 0; i < table.size(); i++)
      assertalways(table(i).size() == n);
   }

/*!
 * \brief Set up pilot sequence for the current frame as given
 */
template <class real, bool norm>
void dminner<real, norm>::copypilot(
      const libbase::vector<libbase::bitfield>& pilot_b)
   {
   validate_bitfield_length(pilot_b);
   ws = pilot_b;
   }

/*!
 * \brief Set up LUT with the given codewords
 */
template <class real, bool norm>
void dminner<real, norm>::copylut(const int i, const libbase::vector<
      libbase::bitfield>& lut_b)
   {
   assertalways(lut_b.size() == num_symbols());
   validate_bitfield_length(lut_b);
   // convert to vector of integers
   const libbase::vector<int> lut_i(lut_b);
   // insert into matrix
   assertalways(i >= 0 && i < num_codebooks());
   assertalways(lut.size().cols() == num_symbols());
   lut.insertrow(lut_i, i);
   }

/*!
 * \brief Display LUT on given stream
 */

template <class real, bool norm>
void dminner<real, norm>::showlut(std::ostream& sout) const
   {
   assert(num_codebooks() >= 1);
   assert(lut.size().cols() == num_symbols());
   for (int i = 0; i < num_codebooks(); i++)
      {
      sout << "LUT " << i << " (k=" << k << ", n=" << n << "):" << std::endl;
      for (int d = 0; d < num_symbols(); d++)
         sout << d << "\t" << libbase::bitfield(lut(i, d), n) << "\t"
               << libbase::weight(lut(i, d)) << std::endl;
      }
   }

/*!
 * \brief Confirm that LUT is valid
 * Checks that all LUT entries are within range and that there are no
 * duplicate entries.
 */

template <class real, bool norm>
void dminner<real, norm>::validatelut() const
   {
   assertalways(num_codebooks() >= 1);
   assertalways(lut.size().cols() == num_symbols());
   for (int i = 0; i < num_codebooks(); i++)
      for (int d = 0; d < num_symbols(); d++)
         {
         // all entries should be within size
         assertalways(lut(i,d) >= 0 && lut(i,d) < (1<<n));
         // all entries should be distinct for each LUT index
         for (int dd = 0; dd < d; dd++)
            assertalways(lut(i, dd) != lut(i, d));
         }
   }

//! Compute and update mean density of sparse alphabet

template <class real, bool norm>
void dminner<real, norm>::computemeandensity()
   {
   array2i_t w = lut;
   w.apply(libbase::weight);
   f = w.sum() / double(n * w.size());
#ifndef NDEBUG
   if (n > 2)
      libbase::trace << "Watermark code density = " << f << std::endl;
#endif
   }

//! Inform user if I or xmax have changed

template <class real, bool norm>
void dminner<real, norm>::checkforchanges(int I, int xmax) const
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

template <class real, bool norm>
void dminner<real, norm>::work_results(const array1b_t& r, array1vr_t& ptable,
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
   const int N = ws.size();
   // Initialise result vector (one sparse symbol per timestep)
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
template <class real, bool norm>
void dminner<real, norm>::normalize_results(const array1vr_t& in,
      array1vd_t& out) const
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

template <class real, bool norm>
void dminner<real, norm>::init()
   {
   // Fill default LUT if necessary
   if (lut_type == lut_straight)
      {
      libbase::sparse codebook(1 << k, n);
      lut.init(1, num_symbols());
      lut.insertrow(array1i_t(codebook), 0);
      }
#ifndef NDEBUG
   // Display LUT when debugging
   if (n > 2)
      showlut(libbase::trace);
#endif
   // Validate LUT and compute the mean density
   validatelut();
   computemeandensity();
   // set default thresholds if necessary
   if (!user_threshold)
      {
      th_inner = real(1e-15);
      th_outer = real(1e-6);
      }
   // Seed the watermark generator and clear the sequence
   r.seed(0);
   ws.init(0);
   // Check that everything makes sense
   test_invariant();
   }

// constructor / destructor

template <class real, bool norm>
dminner<real, norm>::dminner(const int n, const int k) :
   n(n), k(k), lut_type(lut_straight), user_threshold(false)
   {
   init();
   }

template <class real, bool norm>
dminner<real, norm>::dminner(const int n, const int k, const double th_inner,
      const double th_outer) :
   n(n), k(k), lut_type(lut_straight), user_threshold(true), th_inner(real(
         th_inner)), th_outer(real(th_outer))
   {
   init();
   }

// Watermark-specific setup functions

/*!
 * \copydoc set_pilot()
 * \todo Consider moving this method to the dminner2d class
 */
template <class real, bool norm>
void dminner<real, norm>::set_pilot(libbase::vector<bool> pilot)
   {
   assertalways((pilot.size() % n) == 0);
   // init space for converted vector
   libbase::vector<libbase::bitfield> pilot_b(pilot.size() / n);
   // convert pilot sequence
   for (int i = 0; i < pilot_b.size(); i++)
      pilot_b(i) = libbase::bitfield(pilot.extract(i * n, n));
   // pass through the standard method for setting pilot sequence
   set_pilot(pilot_b);
   }

/*!
 * \brief Overrides the internally-generated pilot sequence with given one
 * 
 * The intent of this method is to allow users to apply the dminner decoder
 * in derived algorithms, such as the 2D extension.
 * 
 * \todo merge with copypilot()
 */
template <class real, bool norm>
void dminner<real, norm>::set_pilot(libbase::vector<libbase::bitfield> pilot)
   {
   copypilot(pilot);
   }

/*!
 * \brief Overrides the sparse alphabet with given one
 * 
 * The intent of this method is to allow users to apply the dminner decoder
 * in derived algorithms, such as the 2D extension.
 */
template <class real, bool norm>
void dminner<real, norm>::set_lut(libbase::vector<libbase::bitfield> lut_b)
   {
   // allocate memory and copy read LUT
   lut.init(1, num_symbols());
   copylut(0, lut_b);
   // update LUT-dependent values
   computemeandensity();
#if DEBUG>=2
   showlut(libbase::trace);
#endif
   }

template <class real, bool norm>
void dminner<real, norm>::set_thresholds(const real th_inner,
      const real th_outer)
   {
   user_threshold = true;
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   test_invariant();
   }

// implementations of channel-specific metrics for fba

template <class real, bool norm>
real dminner<real, norm>::R(const int i, const array1b_t& r)
   {
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   const int word = i / n;
   const int bit = i % n;
   bool t = ((ws(word) >> bit) & 1);
   // compute the conditional probability
   return real(mychan.receive(t, r));
   }

// block advance operation - update watermark sequence

template <class real, bool norm>
void dminner<real, norm>::advance() const
   {
   // Inherit sizes
   const int tau = this->input_block_size();
   // Advance pilot sequence only for non-zero block sizes
   if (tau > 0)
      {
      // Initialize space
      ws.init(tau);
      switch (ws_type)
         {
         case ws_random:
            // creates 'tau' elements of 'n' bits each
            for (int i = 0; i < tau; i++)
               ws(i) = r.ival(1 << n);
            break;

         case ws_zero:
            ws = 0;
            break;

         case ws_alt_symbol:
            // alternating all-ones and all-zeros for successive symbols
            for (int i = 0; i < tau; i++)
               ws(i) = (i % 2) == 0 ? 0 : (1 << n) - 1;
            break;

         case ws_mod_vec:
            // repeated modification vectors from list
            for (int i = 0; i < tau; i++)
               ws(i) = ws_vectors(i % ws_vectors.size());
            break;

         default:
            failwith("Unknown watermark sequence type");
            break;
         }
      }
   }

// encoding and decoding functions

template <class real, bool norm>
void dminner<real, norm>::domodulate(const int N, const array1i_t& encoded,
      array1b_t& tx)
   {
   // TODO: when N is removed from the interface, rename 'tau' to 'N'
   // Inherit sizes
   const int q = 1 << k;
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == encoded.size());
   // Each 'encoded' symbol must be representable by a single sparse vector
   assertalways(N == q);
   // Initialise result vector (one bit per sparse vector)
   tx.init(n * tau);
   assertalways(ws.size() == tau);
   // Encode source stream
   for (int i = 0; i < tau; i++)
      tx.segment(i * n, n) = encode(i, encoded(i));
   }

template <class real, bool norm>
void dminner<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, array1vd_t& ptable)
   {
   // Inherit sizes
   const int N = this->input_block_size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   mychan = dynamic_cast<const bsid&> (chan);
   // Update substitution probability to take into account sparse addition
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
   FBA::init(tau, I, xmax, th_inner);
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

template <class real, bool norm>
void dminner<real, norm>::dodemodulate(const channel<bool>& chan,
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

template <class real, bool norm>
void dminner<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const libbase::size_type<
            libbase::vector> offset)
   {
   failwith("Function not implemented.");
   }

// description output

template <class real, bool norm>
std::string dminner<real, norm>::description() const
   {
   std::ostringstream sout;
   sout << "DM Inner Code (" << n << "/" << k << ", ";
   switch (lut_type)
      {
      case lut_straight:
         sout << "sequential codebook";
         break;

      case lut_user:
         sout << lutname << " codebook";
         break;

      case lut_tvb:
         sout << lutname << " codebook [TVB]";
         break;

      default:
         failwith("Unknown LUT type");
         break;
      }
   if (user_threshold)
      sout << ", thresholds " << th_inner << "/" << th_outer;
   if (norm)
      sout << ", normalized";
   switch (ws_type)
      {
      case ws_random:
         sout << ", random watermark";
         break;

      case ws_zero:
         sout << ", no watermark";
         break;

      case ws_alt_symbol:
         sout << ", symbol-alternating watermark";
         break;

      case ws_mod_vec:
         sout << ", modification vectors (length " << ws_vectors.size() << ")";
         break;

      default:
         failwith("Unknown watermark sequence type");
         break;
      }
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real, bool norm>
std::ostream& dminner<real, norm>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 2 << std::endl;
   sout << "# User threshold?" << std::endl;
   sout << user_threshold << std::endl;
   if (user_threshold)
      {
      sout << "#: Inner threshold" << std::endl;
      sout << th_inner << std::endl;
      sout << "#: Outer threshold" << std::endl;
      sout << th_outer << std::endl;
      }
   sout << "# n" << std::endl;
   sout << n << std::endl;
   sout << "# k" << std::endl;
   sout << k << std::endl;
   sout << "# LUT type (0=sparse, 1=user, 2=tvb)" << std::endl;
   sout << lut_type << std::endl;
   switch (lut_type)
      {
      case lut_straight:
         break;

      case lut_user:
         sout << "#: LUT name" << std::endl;
         sout << lutname << std::endl;
         sout << "#: LUT entries" << std::endl;
         assert(num_codebooks() == 1);
         assert(lut.size().cols() == num_symbols());
         for (int d = 0; d < num_symbols(); d++)
            sout << libbase::bitfield(lut(0, d), n) << std::endl;
         break;

      case lut_tvb:
         sout << "#: LUT name" << std::endl;
         sout << lutname << std::endl;
         sout << "#: LUT count" << std::endl;
         sout << num_codebooks() << std::endl;
         assert(num_codebooks() >= 1);
         assert(lut.size().cols() == num_symbols());
         for (int i = 0; i < num_codebooks(); i++)
            {
            sout << "#: LUT entries (table " << i << ")" << std::endl;
            for (int d = 0; d < num_symbols(); d++)
               sout << libbase::bitfield(lut(i, d), n) << std::endl;
            }
         break;

      default:
         failwith("Unknown LUT type");
         break;
      }
   sout << "# WS type (0=random, 1=zero, 2=symbol-alternating, 3=mod-vectors)"
         << std::endl;
   sout << ws_type << std::endl;
   if (ws_type == ws_mod_vec)
      {
      sout << "#: WS modification vectors" << std::endl;
      sout << ws_vectors.size() << std::endl;
      for (int i = 0; i < ws_vectors.size(); i++)
         sout << libbase::bitfield(ws_vectors(i), n) << std::endl;
      }
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering
 *
 * \version 2 Added watermark sequence type
 */

template <class real, bool norm>
std::istream& dminner<real, norm>::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // handle old-format files (without version number)
   if (version < 2)
      {
      //sin.clear();
      sin.seekg(start);
      version = 1;
      }
   sin >> libbase::eatcomments >> user_threshold;
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
      sin >> libbase::eatcomments >> th_inner;
      sin >> libbase::eatcomments >> th_outer;
      }
   sin >> libbase::eatcomments >> n;
   sin >> libbase::eatcomments >> k;
   int temp;
   sin >> libbase::eatcomments >> temp;
   lut_type = (lut_t) temp;
   switch (lut_type)
      {
      case lut_straight:
         break;

      case lut_user:
         {
         sin >> libbase::eatcomments >> lutname;
         // allocate memory
         lut.init(1, num_symbols());
         // read LUT from stream
         libbase::vector<libbase::bitfield> lut_b;
         lut_b.init(num_symbols());
         sin >> libbase::eatcomments;
         lut_b.serialize(sin);
         // copy read LUT
         copylut(0, lut_b);
         }
         break;

      case lut_tvb:
         sin >> libbase::eatcomments >> lutname;
         // read LUT count
         sin >> libbase::eatcomments >> temp;
         // allocate memory
         lut.init(temp, num_symbols());
         for (int i = 0; i < num_codebooks(); i++)
            {
            // read LUT from stream
            libbase::vector<libbase::bitfield> lut_b;
            lut_b.init(num_symbols());
            sin >> libbase::eatcomments;
            lut_b.serialize(sin);
            // copy read LUT
            copylut(i, lut_b);
            }
         break;

      default:
         failwith("Unknown LUT type");
         break;
      }
   // read watermark sequence type if present
   if (version < 2)
      ws_type = ws_random;
   else
      {
      int temp;
      sin >> libbase::eatcomments >> temp;
      ws_type = (ws_t) temp;
      if (ws_type == ws_mod_vec)
         {
         // read WS modification vectors from stream
         libbase::vector<libbase::bitfield> ws_vectors_b;
         sin >> libbase::eatcomments >> ws_vectors_b;
         // copy list of modification vectors
         validate_bitfield_length(ws_vectors_b);
         ws_vectors = ws_vectors_b;
         }
      }
   init();
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner<logrealfast, false> ;
template <>
const serializer dminner<logrealfast, false>::shelper =
      serializer("blockmodem", "dminner<logrealfast>", dminner<logrealfast,
            false>::create);

template class dminner<double, true> ;
template <>
const serializer dminner<double, true>::shelper = serializer("blockmodem",
      "dminner<double>", dminner<double, true>::create);

template class dminner<float, true> ;
template <>
const serializer dminner<float, true>::shelper = serializer("blockmodem",
      "dminner<float>", dminner<float, true>::create);

} // end namespace
