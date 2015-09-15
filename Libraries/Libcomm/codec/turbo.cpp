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

#include "turbo.h"
#include "interleaver/lut/flat.h"
#include "vectorutils.h"
#include <sstream>
#include <iomanip>

namespace libcomm {

// initialization / de-allocation

template <class real, class dbl>
void turbo<real, dbl>::init()
   {
   // check presence and size of interleavers
   assertalways(inter.size() > 0);
   for (int i = 0; i < inter.size(); i++)
      {
      assertalways(inter(i));
      assertalways(inter(i)->size() == inter(0)->size());
      libbase::trace << "Interleaver " << i << ": " << inter(i)->description()
            << std::endl;
      }

   // check required components and initialize BCJR
   assertalways(encoder);
   const int tau = num_timesteps();
   assertalways(tau > 0);
   BCJR::init(*encoder, tau);

   assertalways(!endatzero || !circular);
   assertalways(iter > 0);

   initialised = false;
   }

template <class real, class dbl>
void turbo<real, dbl>::free()
   {
   if (encoder != NULL)
      delete encoder;
   for (int i = 0; i < inter.size(); i++)
      delete inter(i);
   }

template <class real, class dbl>
void turbo<real, dbl>::reset()
   {
   if (circular)
      {
      libbase::allocate(ss, num_sets(), enc_states());
      libbase::allocate(se, num_sets(), enc_states());
      ss = dbl(1.0 / double(enc_states()));
      se = dbl(1.0 / double(enc_states()));
      }
   else if (endatzero)
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

// constructor / destructor

template <class real, class dbl>
turbo<real, dbl>::turbo()
   {
   encoder = NULL;
   }

template <class real, class dbl>
turbo<real, dbl>::turbo(const fsm& encoder, const libbase::vector<interleaver<
      dbl> *>& inter, const int iter, const bool endatzero,
      const bool parallel, const bool circular)
   {
   this->encoder = dynamic_cast<fsm*> (encoder.clone());
   this->inter = inter;
   this->endatzero = endatzero;
   this->parallel = parallel;
   this->circular = circular;
   this->iter = iter;
   init();
   }

// memory allocator (for internal use only)

template <class real, class dbl>
void turbo<real, dbl>::allocate()
   {
   // Inherit sizes
   const int sets = num_sets();
   const int tau = num_timesteps();
   const int K = alg_input_symbols();
   const int N = alg_output_symbols();

   rp.init(tau, K);
   if (parallel)
      libbase::allocate(ra, sets, tau, K);
   else
      libbase::allocate(ra, 1, tau, K);
   libbase::allocate(R, sets, tau, N);
   // flag the state of the arrays
   initialised = true;

   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const std::streamsize prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(dbl) * (rp.size() + ra.size()
         * ra(0).size() + R.size() * R(0).size());
   std::cerr << "Turbo Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
   }

// wrapping functions

/*!
 * \brief Computes extrinsic probabilities
 * \param[in]  ra  A-priori (extrinsic) probabilities of input values
 * \param[in]  ri  A-posteriori probabilities of input values
 * \param[in]  r   A-priori intrinsic probabilities of input values
 * \param[out] re  Extrinsic probabilities of input values
 *
 * \note It is counter-productive to vectorize this, as it would require
 * many unnecessary temporary matrix creations.
 *
 * \warning The return matrix re may actually be one of the input matrices,
 * so one must be careful not to overwrite positions that still
 * need to be read.
 */
template <class real, class dbl>
void turbo<real, dbl>::work_extrinsic(const array2d_t& ra, const array2d_t& ri,
      const array2d_t& r, array2d_t& re)
   {
   // Compute denominator
   array2d_t rar = ra.multiply(r);
   // Copy numerator and divide only for non-zero denominator
   re = ri;
   re.mask(rar > 0).divideby(rar);
   }

/*!
 * \brief Complete BCJR decoding cycle
 * \param[in]  set Parity sequence being decoded
 * \param[in]  ra  A-priori (extrinsic) probabilities of input values
 * \param[out] ri  A-posteriori probabilities of input values
 * \param[out] re  Extrinsic probabilities of input values (will be used later
 * as the new 'a-priori' probabilities)
 *
 * This method performs a complete decoding cycle, including start/end state
 * probability settings for circular decoding, and any interleaving/de-
 * interleaving.
 *
 * \note When using a circular trellis, the start- and end-state probabilities
 * are re-initialize with the stored values from the previous turn.
 *
 * \warning The return matrix re may actually be the input matrix ra,
 * so one must be careful not to overwrite positions that still
 * need to be read.
 */
template <class real, class dbl>
void turbo<real, dbl>::bcjr_wrap(const int set, const array2d_t& ra,
      array2d_t& ri, array2d_t& re)
   {
   // Temporary variables to hold interleaved versions of ra/ri
   array2d_t rai, rii;
   if (circular)
      {
      BCJR::setstart(ss(set));
      BCJR::setend(se(set));
      }
   inter(set)->transform(ra, rai);
   BCJR::fdecode(R(set), rai, rii);
   inter(set)->inverse(rii, ri);
   if (circular)
      {
      ss(set) = BCJR::getstart();
      se(set) = BCJR::getend();
      }
   work_extrinsic(ra, ri, rp, re);
   }

/*! \brief Perform a complete serial-decoding cycle
 *
 * \note The BCJR normalization method is used to normalize the extrinsic
 * probabilities.
 */
template <class real, class dbl>
void turbo<real, dbl>::decode_serial(array2d_t& ri)
   {
   // after working all sets, ri is the intrinsic+extrinsic information
   // from the last stage decoder.
   for (int set = 0; set < num_sets(); set++)
      {
      bcjr_wrap(set, ra(0), ri, ra(0));
      BCJR::normalize(ra(0));
      }
   BCJR::normalize(ri);
   }

/*! \brief Perform a complete parallel-decoding cycle
 *
 * \note The BCJR normalization method is used to normalize the extrinsic
 * probabilities.
 *
 * \warning Simulations show that parallel-decoding works well with the
 * 1/4-rate, 3-code, K=3 (111/101), N=4096 code from divs95b;
 * however, when simulating larger codes (N=8192) the system seems
 * to go unstable after a few iterations. Also significantly, similar
 * codes with lower rates (1/6 and 1/8) perform _worse_ as the rate
 * decreases.
 */
template <class real, class dbl>
void turbo<real, dbl>::decode_parallel(array2d_t& ri)
   {
   // here ri is only a temporary space
   // and ra(set) is updated with the extrinsic information for that set
   for (int set = 0; set < num_sets(); set++)
      bcjr_wrap(set, ra(set), ri, ra(set));
   // the following are repeated at each frame element, for each possible symbol
   // work in ri the sum of all extrinsic information
   ri = ra(0);
   for (int set = 1; set < num_sets(); set++)
      ri.multiplyby(ra(set));
   // compute the next-stage a priori information by subtracting the extrinsic
   // information of the current stage from the sum of all extrinsic information.
   for (int set = 0; set < num_sets(); set++)
      ra(set) = ri.divide(ra(set));
   // add the channel information to the sum of extrinsic information
   ri.multiplyby(rp);
   // normalize results
   for (int set = 0; set < num_sets(); set++)
      BCJR::normalize(ra(set));
   BCJR::normalize(ri);
   }

/*! \copydoc codec_softout::setreceiver()
 *
 * Sets: rp, ra, R, [ss, se, through reset()]
 *
 * \note The BCJR normalization method is used to normalize the channel-derived
 * (intrinsic) probabilities 'r' and 'R'; in view of this, the a-priori
 * probabilities are now created normalized.
 *
 * \note Clean up this function, removing unnecessary symbol-conversion
 */
template <class real, class dbl>
void turbo<real, dbl>::do_init_decoder(const array1vd_t& ptable)
   {
   assert(ptable.size() == This::output_block_size());
   // Inherit sizes
   const int sets = num_sets();
   const int tau = num_timesteps();
   const int k = enc_inputs();
   const int p = enc_parity();
   const int S = enc_symbols();
   const int K = alg_input_symbols();
   const int N = alg_output_symbols();
   // Derived sizes
   const int s = k + p * sets;
   const int P = N / K;

   // initialise memory if necessary
   if (!initialised)
      allocate();

   // Allocate space for temporary matrices
   libbase::matrix3<dbl> ptemp(sets, tau, P);

   // Get the necessary data from the channel
   for (int t = 0; t < tau; t++)
      {
      // Input (data) bits [set 0 only]
      for (int x = 0; x < K; x++)
         {
         rp(t, x) = 1;
         for (int i = 0, thisx = x; i < k; i++, thisx /= S)
            rp(t, x) *= dbl(ptable(t * s + i)(thisx % S));
         }
      // Parity bits [all sets]
      for (int x = 0; x < P; x++)
         for (int set = 0, offset = k; set < sets; set++)
            {
            ptemp(set, t, x) = 1;
            for (int i = 0, thisx = x; i < p; i++, thisx /= S)
               ptemp(set, t, x) *= dbl(ptable(t * s + i + offset)(thisx % S));
            offset += p;
            }
      }

   // Initialise a priori probabilities (extrinsic)
   for (int set = 0; set < (parallel ? sets : 1); set++)
      ra(set) = 1.0;

   // Normalize a priori probabilities (intrinsic - source)
   BCJR::normalize(rp);

   // Compute and normalize a priori probabilities (intrinsic - encoded)
   array2d_t rpi;
   for (int set = 0; set < sets; set++)
      {
      inter(set)->transform(rp, rpi);
      for (int t = 0; t < tau; t++)
         for (int x = 0; x < N; x++)
            R(set)(t, x) = rpi(t, x % K) * ptemp(set, t, x / K);
      BCJR::normalize(R(set));
      }

   // Reset start- and end-state probabilities
   reset();
   }

template <class real, class dbl>
void turbo<real, dbl>::do_init_decoder(const array1vd_t& ptable, const array1vd_t& app)
   {
   // Start by setting receiver statistics
   do_init_decoder(ptable);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(app.size() > 0);
   assertalways(app(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(app.size() == This::input_block_size());
   // Copy the input statistics for the BCJR Algorithm
   for (int t = 0; t < rp.size().rows(); t++)
      for (int i = 0; i < rp.size().cols(); i++)
         rp(t, i) *= app(t)(i);
   }

// encoding and decoding functions

template <class real, class dbl>
void turbo<real, dbl>::seedfrom(libbase::random& r)
   {
   for (int set = 0; set < num_sets(); set++)
      inter(set)->seedfrom(r);
   }

template <class real, class dbl>
void turbo<real, dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Inherit sizes
   const int sets = num_sets();
   const int tau = num_timesteps();
   const int k = enc_inputs();
   const int p = enc_parity();
   const int S = enc_symbols();
   // Derived sizes
   const int s = k + p * sets;

   // Reform source into a matrix, with one row per timestep
   // and adding any necessary tail
   array2i_t source1(tau, k);
   source1 = fsm::tail;
   source1.copyfrom(source);

   // Declare space for the interleaved source
   array2i_t source2(tau, k);
   // Allocate space for the encoder outputs
   libbase::matrix<libbase::vector<int> > x(sets, tau);
   // Consider sets in order
   for (int set = 0; set < sets; set++)
      {
      // Create interleaved version of source
      for (int i = 0; i < k; i++)
         {
         array1i_t source2slice;
         inter(set)->transform(source1.extractcol(i), source2slice);
         source2.insertcol(source2slice, i);
         }

      // Reset the encoder to zero state
      encoder->reset();

      // When dealing with a circular system, perform first pass to determine
      // end state, then reset to the corresponding circular state.
      int cstate = 0;
      if (circular)
         {
         for (int t = 0; t < tau; t++)
            {
            array1i_t ip = source2.extractrow(t);
            encoder->advance(ip);
            }
         encoder->resetcircular();
         cstate = fsm::convert(encoder->state(), S);
         }

      // Encode source
      // (non-interleaved must be done first to determine tail bit values)
      for (int t = 0; t < tau; t++)
         {
         array1i_t ip = source2.extractrow(t);
         x(set, t) = encoder->step(ip).extract(k, p);
         source2.insertrow(ip, t);
         }

      // If this was the first (non-interleaved) set, copy back the source
      // to fix the tail bit values, if any
      if (endatzero && set == 0)
         source1 = source2;

      // check that encoder finishes correctly
      const int finstate = fsm::convert(encoder->state(), S);
      if (circular)
         assertalways(finstate == cstate);
      if (endatzero)
         assertalways(finstate == 0);
      }

   // Initialise result vector
   encoded.init(This::output_block_size());
   // Encode source stream
   for (int t = 0; t < tau; t++)
      {
      // data bits
      encoded.segment(t * s, k) = source1.extractrow(t);
      // parity bits
      for (int set = 0; set < sets; set++)
         encoded.segment(t * s + k + p * set, p) = x(set, t);
      }
   }

template <class real, class dbl>
void turbo<real, dbl>::softdecode(array1vd_t& ri)
   {
   // temporary space to hold complete results (ie. with tail)
   array2d_t rif;
   // do one iteration, in serial or parallel as required
   if (parallel)
      decode_parallel(rif);
   else
      decode_serial(rif);
   // remove any tail bits from input set
   libbase::allocate(ri, input_block_size(), num_inputs());
   for (int i = 0; i < input_block_size(); i++)
      for (int j = 0; j < num_inputs(); j++)
         ri(i)(j) = rif(i, j);
   }

template <class real, class dbl>
void turbo<real, dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   failwith("Not yet implemented");
   }

// description output

template <class real, class dbl>
std::string turbo<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Turbo Code (" << This::output_bits() << "," << This::input_bits()
         << ") - ";
   sout << encoder->description() << ", ";
   for (int i = 0; i < inter.size(); i++)
      sout << inter(i)->description() << ", ";
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << (parallel ? "Parallel Decoding, " : "Serial Decoding, ");
   sout << iter << " iterations";
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& turbo<real, dbl>::serialize(std::ostream& sout) const
   {
   // format version
   sout << "# Version" << std::endl;
   sout << 2 << std::endl;
   sout << "# Encoder" << std::endl;
   sout << encoder;
   sout << "# Number of parallel sets" << std::endl;
   sout << num_sets() << std::endl;
   for (int i = 0; i < inter.size(); i++)
      {
      sout << "# Interleaver " << i << std::endl;
      sout << inter(i);
      }
   sout << "# Terminated?" << std::endl;
   sout << int(endatzero) << std::endl;
   sout << "# Circular?" << std::endl;
   sout << int(circular) << std::endl;
   sout << "# Parallel decoder?" << std::endl;
   sout << int(parallel) << std::endl;
   sout << "# Number of iterations" << std::endl;
   sout << iter << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering; added explicit first interleaver
 *
 * \version 2 Removed explicit 'tau'
 */
template <class real, class dbl>
std::istream& turbo<real, dbl>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // handle old-format files
   if (sin.fail())
      {
      version = 0;
      sin.clear();
      }
   sin >> libbase::eatcomments >> encoder >> libbase::verify;
   int tau = 0;
   if (version < 2)
      sin >> libbase::eatcomments >> tau >> libbase::verify;
   int sets;
   sin >> libbase::eatcomments >> sets >> libbase::verify;
   inter.init(sets);
   if (version < 1)
      {
      inter(0) = new flat<dbl> (tau);
      for (int i = 1; i < inter.size(); i++)
         sin >> libbase::eatcomments >> inter(i) >> libbase::verify;
      }
   else
      {
      for (int i = 0; i < inter.size(); i++)
         sin >> libbase::eatcomments >> inter(i) >> libbase::verify;
      }
   sin >> libbase::eatcomments >> endatzero >> libbase::verify;
   sin >> libbase::eatcomments >> circular >> libbase::verify;
   sin >> libbase::eatcomments >> parallel >> libbase::verify;
   sin >> libbase::eatcomments >> iter >> libbase::verify;
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

/* Serialization string: turbo<real1,real2>
 * where:
 *      real1 = float | double | mpreal | mpgnu | logreal | logrealfast
 *              [real1 is the internal arithmetic type]
 *      real2 = float | double | logrealfast
 *              [real2 is the inter-iteration statistics type]
 */
#define INSTANTIATE(r, args) \
      template class turbo<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer turbo<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "codec", \
            "turbo<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            turbo<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (REAL1_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
