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

#include "dminner2.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show prior and posterior sof/eof probabilities when decoding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Setup procedure

template <class real, bool norm>
void dminner2<real, norm>::init(const channel<bool>& chan,
      const array1d_t& sof_pdf, const int offset)
   {
   // Inherit block size from last modulation step
   const int q = 1 << Base::k;
   const int n = Base::n;
   const int N = Base::ws.size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   Base::mychan = dynamic_cast<const bsid&> (chan);
   // Set channel block size to q-ary symbol size
   Base::mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const int I = Base::mychan.compute_I(tau);
   const int xmax = Base::mychan.compute_xmax(tau, sof_pdf, offset);
   const int dxmax = Base::mychan.compute_xmax(n);
   Base::checkforchanges(I, xmax);
   // Initialize forward-backward algorithm
   fba.init(N, n, q, I, xmax, dxmax, Base::th_inner, Base::th_outer);
   // initialize our embedded metric computer with unchanging elements
   fba.get_receiver().init(n, Base::lut, Base::mychan);
   }

template <class real, bool norm>
void dminner2<real, norm>::advance() const
   {
   // advance the base class
   Base::advance();
   // initialize our embedded metric computer
   fba.get_receiver().init(Base::ws);
   }

// encoding and decoding functions

template <class real, bool norm>
void dminner2<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, array1vd_t& ptable)
   {
   const array1vd_t app; // empty APP table
   dodemodulate(chan, rx, app, ptable);
   }

template <class real, bool norm>
void dminner2<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   // Initialize for known-start
   init(chan);
   // Shorthand for transmitted and received frame sizes
   const int tau = this->output_block_size();
   const int rho = rx.size();
   // Algorithm parameters
   const int xmax = fba.get_xmax();
   // Check that rx size is within valid range
   assertalways(xmax >= abs(rho - tau));
   // Set up start-of-frame drift pdf (drift = 0)
   array1d_t sof_prior;
   sof_prior.init(2 * xmax + 1);
   sof_prior = 0;
   sof_prior(xmax + 0) = 1;
   // Set up end-of-frame drift pdf (drift = rho-tau)
   array1d_t eof_prior;
   eof_prior.init(2 * xmax + 1);
   eof_prior = 0;
   eof_prior(xmax + rho - tau) = 1;
   // Offset rx by xmax and pad to a total size of tau+2*xmax
   array1b_t r;
   r.init(tau + 2 * xmax);
   r.segment(xmax, rho) = rx;
   // Delegate
   array1d_t sof_post;
   array1d_t eof_post;
   demodulate_wrapper(chan, r, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, libbase::size_type<libbase::vector>(xmax));
   }

template <class real, bool norm>
void dminner2<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const libbase::size_type<
            libbase::vector> offset)
   {
   // Initialize for known-start
   init(chan, sof_prior, offset);
   // TODO: validate priors have required size?
#ifndef NDEBUG
   std::cerr << "DEBUG (dminner2): offset = " << offset << ", xmax = "
         << fba.get_xmax() << "." << std::endl;
#endif
   assert(offset == fba.get_xmax());
   // Delegate
   demodulate_wrapper(chan, rx, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, offset);
   }

/*!
 * \brief Wrapper for calling demodulation algorithm
 *
 * This method assumes that the init() method has already been called with
 * the appropriate parameters.
 */
template <class real, bool norm>
void dminner2<real, norm>::demodulate_wrapper(const channel<bool>& chan,
      const array1b_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const int offset)
   {
   // Call FBA and normalize results
#if DEBUG>=2
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
#endif
   array1vr_t ptable_r;
   array1r_t sof_post_r;
   array1r_t eof_post_r;
   fba.decode(*this, rx, sof_prior, eof_prior, app, ptable_r, sof_post_r,
         eof_post_r, offset);
   Base::normalize_results(ptable_r, ptable);
   normalize(sof_post_r, sof_post);
   normalize(eof_post_r, eof_post);
#if DEBUG>=2
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
#endif
   }

/*!
 * \brief Normalize probability table
 *
 * The input probability table is normalized such that the largest value is
 * equal to 1; result is converted to double.
 */
template <class real, bool norm>
void dminner2<real, norm>::normalize(const array1r_t& in, array1d_t& out)
   {
   const int N = in.size();
   assert(N > 0);
   // check for numerical underflow
   real scale = in.max();
   assert(scale != real(0));
   scale = real(1) / scale;
   // allocate result space
   out.init(N);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      out(i) = in(i) * scale;
   }

// description output

template <class real, bool norm>
std::string dminner2<real, norm>::description() const
   {
   std::ostringstream sout;
   sout << "Symbol-level " << Base::description();
   sout << ", " << fba.description();
   return sout.str();
   }

// object serialization - saving

template <class real, bool norm>
std::ostream& dminner2<real, norm>::serialize(std::ostream& sout) const
   {
   return Base::serialize(sout);
   }

// object serialization - loading

template <class real, bool norm>
std::istream& dminner2<real, norm>::serialize(std::istream& sin)
   {
   return Base::serialize(sin);
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;
using libbase::serializer;

#ifndef USE_CUDA
template class dminner2<logrealfast, false> ;
template <>
const serializer dminner2<logrealfast, false>::shelper = serializer(
      "blockmodem", "dminner2<logrealfast>",
      dminner2<logrealfast, false>::create);
#endif

template class dminner2<double, true> ;
template <>
const serializer dminner2<double, true>::shelper = serializer("blockmodem",
      "dminner2<double>", dminner2<double, true>::create);

template class dminner2<float, true> ;
template <>
const serializer dminner2<float, true>::shelper = serializer("blockmodem",
      "dminner2<float>", dminner2<float, true>::create);

} // end namespace
