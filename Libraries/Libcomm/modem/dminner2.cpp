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

template <class real>
void dminner2<real>::init(const channel<bool>& chan, const array1d_t& sof_pdf,
      const int offset)
   {
   // Inherit block size from last modulation step
   const int q = 1 << Base::k;
   const int n = Base::n;
   const int N = Base::marker.size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   Base::mychan = dynamic_cast<const bsid&> (chan);
   // Set channel block size to q-ary symbol size
   Base::mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const int I = Base::mychan.compute_I(tau);
   // No need to recompute xmax if we are given a prior PDF
   const int xmax = sof_pdf.size() > 0 ? offset : Base::mychan.compute_xmax(
         tau, sof_pdf, offset);
   const int dxmax = Base::mychan.compute_xmax(n);
   Base::checkforchanges(I, xmax);
   // Initialize forward-backward algorithm
   fba.init(N, n, q, I, xmax, dxmax, Base::th_inner, Base::th_outer,
         Base::norm, batch, lazy, globalstore);
   // initialize our embedded metric computer with unchanging elements
   fba.get_receiver().init(n, Base::codebook, Base::mychan);
   }

template <class real>
void dminner2<real>::advance() const
   {
   // advance the base class
   Base::advance();
   // initialize our embedded metric computer
   fba.get_receiver().init(Base::marker);
   }

// encoding and decoding functions

template <class real>
void dminner2<real>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, array1vd_t& ptable)
   {
   const array1vd_t app; // empty APP table
   dodemodulate(chan, rx, app, ptable);
   }

template <class real>
void dminner2<real>::dodemodulate(const channel<bool>& chan,
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

template <class real>
void dminner2<real>::dodemodulate(const channel<bool>& chan,
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
template <class real>
void dminner2<real>::demodulate_wrapper(const channel<bool>& chan,
      const array1b_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const int offset)
   {
   // Call FBA and normalize results
#if DEBUG>=2
   using libbase::index_of_max;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "max at " << index_of_max(sof_prior) - offset << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
   std::cerr << "max at " << index_of_max(eof_prior) - offset << std::endl;
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
   std::cerr << "max at " << index_of_max(sof_post) - offset << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
   std::cerr << "max at " << index_of_max(eof_post) - offset << std::endl;
#endif
   }

/*!
 * \brief Normalize probability table
 *
 * The input probability table is normalized such that the largest value is
 * equal to 1; result is converted to double.
 */
template <class real>
void dminner2<real>::normalize(const array1r_t& in, array1d_t& out)
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

template <class real>
std::string dminner2<real>::description() const
   {
   std::ostringstream sout;
   sout << "Symbol-level " << Base::description();
   sout.seekp(-1, std::ios_base::cur);
   if (batch)
      sout << ", batch interface";
   else
      sout << ", single interface";
   if (lazy)
      {
      sout << ", lazy computation";
      if (globalstore)
         sout << ", global caching";
      else
         sout << ", local caching";
      }
   else
      {
      sout << ", pre-computation";
      if (globalstore)
         sout << ", global";
      else
         sout << ", local";
      }
   sout << "), " << fba.description();
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& dminner2<real>::serialize(std::ostream& sout) const
   {
   // serialize base object
   Base::serialize(sout);
   sout << "# Version" << std::endl;
   sout << 3 << std::endl;
   sout << "# Use batch receiver computation?" << std::endl;
   sout << batch << std::endl;
   sout << "# Lazy computation of gamma?" << std::endl;
   sout << lazy << std::endl;
   sout << "# Global storage / caching of computed gamma values?" << std::endl;
   sout << globalstore << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered, no extensions)
 *
 * \version 1 Added version numbering
 *
 * \version 2 Added batch, lazy, caching flags; caching is only
 *      specified if lazy is true (otherwise it is meaningless)
 *
 * \version 3 Changed 'caching' flag to 'global store', now also defined for
 *      pre-computation cases
 */

template <class real>
std::istream& dminner2<real>::serialize(std::istream& sin)
   {
   // serialize base object
   Base::serialize(sin);
   // serialize dminner2 extensions
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // handle old-format files (without version number)
   if (sin.fail() || version < 2)
      {
      sin.clear();
      sin.seekg(start);
      version = 1;
      }
   // read decoder parameters
   if (version >= 2)
      {
      sin >> libbase::eatcomments >> batch >> libbase::verify;
      sin >> libbase::eatcomments >> lazy >> libbase::verify;
      if (lazy || version >= 3)
         sin >> libbase::eatcomments >> globalstore >> libbase::verify;
      else
         globalstore = true;
      }
   else
      {
      batch = true;
      lazy = true;
      globalstore = true;
      }
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

/* Serialization string: dminner2<real,norm>
 * where:
 *      real = float | double | logrealfast (CPU only)
 */
#define INSTANTIATE(r, x, type) \
      template class dminner2<type>; \
      template <> \
      const serializer dminner2<type>::shelper( \
            "blockmodem", \
            "dminner2<" BOOST_PP_STRINGIZE(type) ">", \
            dminner2<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
