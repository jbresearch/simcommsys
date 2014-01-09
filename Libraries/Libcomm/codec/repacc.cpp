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
 */

#include "repacc.h"
#include "vectorutils.h"
#include "hard_decision.h"

#include <sstream>
#include <iomanip>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate encoded + decoded output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// initialization / de-allocation

template <class real, class dbl>
void repacc<real, dbl>::init()
   {
   // check presence of components
   assertalways(inter);
   assertalways(acc);
   // check repeat code
   assertalways(This::input_block_size() > 0);
   assertalways(rep.num_inputs() == This::num_inputs());
   // initialize BCJR subsystem for accumulator
   BCJR::init(*acc, This::acc_timesteps());
   // check interleaver size
   assertalways(inter->size() == This::acc_timesteps());
   assertalways(iter > 0);
   // check lower clipping threshold
   assertalways(limitlo >= dbl(0));

   initialised = false;
   }

template <class real, class dbl>
void repacc<real, dbl>::free()
   {
   if (acc != NULL)
      delete acc;
   if (inter != NULL)
      delete inter;
   }

template <class real, class dbl>
void repacc<real, dbl>::reset()
   {
   if (endatzero)
      {
      BCJR::setstart(0);
      BCJR::setend(0);
      }
   else
      {
      BCJR::setstart(0);
      BCJR::setend();
      }
   }

// memory allocator (for internal use only)

template <class real, class dbl>
void repacc<real, dbl>::allocate()
   {
   libbase::allocate(rp, This::input_block_size(), This::num_inputs());
   ra.init(This::acc_timesteps(), acc->num_input_combinations());
   R.init(This::acc_timesteps(), acc->num_output_combinations());
   // flag the state of the arrays
   initialised = true;

   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const std::streamsize prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(dbl) * (rp.size() + ra.size() + R.size());
   std::cerr << "RepAcc Memory Usage: " << bytes_used / double(1 << 20)
         << "MiB" << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
   }

// constructor / destructor

template <class real, class dbl>
repacc<real, dbl>::repacc() :
   inter(NULL), acc(NULL)
   {
   }

// internal codec functions

template <class real, class dbl>
void repacc<real, dbl>::resetpriors()
   {
   // Should be called after setreceivers()
   assertalways(initialised);
   // Initialise intrinsic source statistics (natural)
   rp = 1.0;
   }

template <class real, class dbl>
void repacc<real, dbl>::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Take into account intrinsic source statistics
   rp = ptable;
   }

/*! \copydoc codec_softout::setreceiver()
 *
 * Sets: ra, R
 *
 * \note The BCJR normalization method is used to normalize the channel-derived
 * (intrinsic) probabilities 'r' and 'R'; in view of this, the a-priori
 * probabilities are now created normalized.
 */
template <class real, class dbl>
void repacc<real, dbl>::setreceiver(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());

   // initialise memory if necessary
   if (!initialised)
      allocate();

   // Initialise extrinsic accumulator-input statistics (natural)
   ra = 1.0;
   // Inherit sizes
   const int k = acc->num_inputs();
   const int n = acc->num_outputs();
   const int tau = acc_timesteps();
   const int N = acc->num_output_combinations();
   const int K = acc->num_input_combinations();
   const int S = acc->num_symbols();
   // Calculate internal sizes
   const int p = n - k;
   // Determine intrinsic accumulator-output statistics (interleaved)
   // from the channel
   R = 1.0;
   for (int t = 0; t < tau; t++)
      for (int x = 0; x < N; x++)
         for (int i = 0, thisx = x / K; i < p; i++, thisx /= S)
            R(t, x) *= dbl(ptable(t * p + i)(thisx % S));
   BCJR::normalize(R);

   // Reset start- and end-state probabilities
   reset();
   }

// encoding and decoding functions

template <class real, class dbl>
void repacc<real, dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Inherit sizes
   const int k = acc->num_inputs();
   const int n = acc->num_outputs();
   const int tau = acc_timesteps();
   // Calculate internal sizes
   const int p = n - k;
#if DEBUG>=2
   std::cerr << "Source:" << std::endl;
   source.serialize(std::cerr, '\n');
#endif

   // Compute repeater output
   array1i_t rep0;
   rep.encode(source, rep0);
   // Copy and add any necessary tail
   array1i_t rep1(tau * k);
   rep1.copyfrom(rep0);
   for (int i = rep0.size(); i < rep1.size(); i++)
      rep1(i) = fsm::tail;
   // Create interleaved sequence
   array1i_t rep2;
   inter->transform(rep1, rep2);
#if DEBUG>=2
   std::cerr << "Repeater:" << std::endl;
   rep1.serialize(std::cerr, '\n');
   std::cerr << "Permuter:" << std::endl;
   rep2.serialize(std::cerr, '\n');
#endif

   // Initialise result vector
   encoded.init(This::output_block_size());
   // Reset the encoder to zero state
   acc->reset();
   // Encode sequence
   for (int i = 0; i < tau; i++)
      {
      array1i_t ip = rep2.segment(i * k, k);
      encoded.segment(i * p, p) = acc->step(ip).extract(k, p);
      }
   // check that encoder finishes correctly
   if (endatzero)
      assertalways(fsm::convert(acc->state(), acc->num_symbols()) == 0);
#if DEBUG>=2
   std::cerr << "Accumulator:" << std::endl;
   encoded.serialize(std::cerr, '\n');
#endif
   }

/*! \copydoc codec_softout::softdecode()
 *
 * \note Implements soft-decision decoding according to Alexandre's
 * interpretation:
 * - when computing final output at repetition code, use only extrinsic
 * information from accumulator
 * - when computing extrinsic output at rep code, factor out the input
 * information at that position
 */
template <class real, class dbl>
void repacc<real, dbl>::softdecode(array1vd_t& ri)
   {
   // decode accumulator

   // Temporary variables to hold posterior probabilities and
   // interleaved versions of ra/ri
   array2d_t rif, rai, rii;
   inter->transform(ra, rai);
   BCJR::fdecode(R, rai, rii);
   inter->inverse(rii, rif);
   // compute extrinsic information
   rif.mask(ra > 0).divideby(ra);
   ra = rif;
   // normalize and clip extrinsic information
   BCJR::normalize(ra);
   ra.mask(ra < limitlo) = limitlo;

   // allocate space for interim results
   const int Nr = rep.output_block_size();
   const int q = rep.num_outputs();
   assertalways(ra.size().rows() >= Nr);
   assertalways(ra.size().cols() == q);
   array1vd_t ravd;
   libbase::allocate(ravd, Nr, q);
   // convert interim results
   for (int i = 0; i < Nr; i++)
      for (int x = 0; x < q; x++)
         ravd(i)(x) = ra(i, x);

#if DEBUG>=2
   array1i_t dec;
   hard_decision<libbase::vector, dbl, int> functor;
   functor(ravd,dec);
   libbase::trace << "DEBUG (repacc): ravd = ";
   dec.serialize(libbase::trace, ' ');
#endif

   // decode repetition code (based on extrinsic information only)
   array1vd_t ro;
   rep.init_decoder(ravd, rp);
   rep.softdecode(ri, ro);

#if DEBUG>=2
   functor(ro,dec);
   libbase::trace << "DEBUG (repacc): ro = ";
   dec.serialize(libbase::trace, ' ');
   functor(ri,dec);
   libbase::trace << "DEBUG (repacc): ri = ";
   dec.serialize(libbase::trace, ' ');
#endif

   // compute extrinsic information
   // TODO: figure out how to deal with tail
   for (int i = 0; i < Nr; i++)
      for (int x = 0; x < q; x++)
         if (ra(i, x) > dbl(0))
            ra(i, x) = ro(i)(x) / ra(i, x);
         else
            ra(i, x) = ro(i)(x);

   // normalize and clip extrinsic information
   BCJR::normalize(ra);
   ra.mask(ra < limitlo) = limitlo;
   }

template <class real, class dbl>
void repacc<real, dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   failwith("Not yet implemented");
   }

// description output

template <class real, class dbl>
std::string repacc<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Repeat-Accumulate Code - ";
   sout << rep.description() << ", ";
   sout << acc->description() << ", ";
   sout << inter->description() << ", ";
   sout << iter << " iterations, ";
   sout << (endatzero ? "Terminated" : "Unterminated");
   if (limitlo > dbl(0))
      sout << ", Clipping at " << limitlo;
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& repacc<real, dbl>::serialize(std::ostream& sout) const
   {
   // format version
   sout << "# Version" << std::endl;
   sout << 3 << std::endl;
   sout << "# Repetition codec" << std::endl;
   rep.serialize(sout);
   sout << "# Accumulator" << std::endl;
   sout << acc;
   sout << "# Interleaver" << std::endl;
   sout << inter;
   sout << "# Number of iterations" << std::endl;
   sout << iter << std::endl;
   sout << "# Terminated?" << std::endl;
   sout << int(endatzero) << std::endl;
   sout << "# Lower clipping threshold" << std::endl;
   sout << limitlo << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 3 added clipping threshold (limitlo)
 */
template <class real, class dbl>
std::istream& repacc<real, dbl>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   assertalways(version >= 2);
   // get version 2 items
   rep.serialize(sin);
   sin >> libbase::eatcomments >> acc >> libbase::verify;
   sin >> libbase::eatcomments >> inter >> libbase::verify;
   sin >> libbase::eatcomments >> iter >> libbase::verify;
   sin >> libbase::eatcomments >> endatzero >> libbase::verify;
   // get version 3 items
   if (version >= 3)
      sin >> libbase::eatcomments >> limitlo >> libbase::verify;
   else
      limitlo = 0;
   init();
   assertalways(sin.good());
   return sin;
   }

} // end namespace

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define REAL1_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)
#define REAL2_TYPE_SEQ \
   (float)(double) \
   (logrealfast)

/* Serialization string: repacc<real1,real2>
 * where:
 *      real1 = float | double | mpreal | mpgnu | logreal | logrealfast
 *              [real1 is the internal arithmetic type]
 *      real2 = float | double | logrealfast
 *              [real2 is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class repacc<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer repacc<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "codec", \
            "repacc<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            repacc<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (REAL1_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
