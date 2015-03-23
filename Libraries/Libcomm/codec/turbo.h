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

#ifndef __turbo_h
#define __turbo_h

#include "config.h"
#include "codec_softout.h"
#include "fsm.h"
#include "interleaver.h"
#include "safe_bcjr.h"
#include "itfunc.h"

#include <cstdlib>
#include <cmath>

namespace libcomm {

/*!
 * \brief   Turbo decoding algorithm.
 * \author  Johann Briffa
 *
 * All internal metrics are held as type 'real', which is user-defined. This
 * allows internal working at any required level of accuracy. This is necessary
 * because the internal metrics have a very wide dynamic range, which increases
 * exponentially with block size 'tau'. Actually, the required range is within
 * [1,0), but very large exponents are required. (For BCJR sub-component)
 *
 * Similarly, the parameter 'dbl' defines the numerical representation for
 * inter-iteration statistics. This became necessary for the parallel decoding
 * structure, where the range of extrinsic information is much larger than for
 * serial decoding; furthermore, this range increases with the number of
 * iterations performed.
 *
 * \note Memory is allocated only on first call to demodulate/decode. This
 * reduces memory requirements in cases where classes are instantiated
 * but not actually used for decoding (e.g. in master node on a
 * distributed Monte Carlo simulation)
 *
 * \note Since puncturing is not handled within the codec, for the moment, the
 * modification for stipple puncturing with simile interleavers is not
 * performed.
 *
 * \note Serialization is versioned; for compatibility, earlier versions are
 * interpreted as v.0; a flat interleaver is automatically used for the
 * first encoder in these cases.
 *
 * \todo Fix terminated sequence encoding (currently this implicitly assumes
 * a flat first interleaver)
 *
 * \todo Standardize encoding/decoding of multiple symbols within a larger
 * symbol space; this parallels what was done in ccfsm.
 *
 * \todo Refactor to use external normalization function
 *
 * \todo Refactor: extract work_extrinsic
 *
 * \todo Remove redundant result vector initializations (these should happen
 * on the first call to a function where that vector is used as an
 * output).
 *
 * \todo Split serial and parallel decoding into separate classes.
 *
 * \todo Update decoding process for changes in FSM model.
 */

template <class real, class dbl>
class turbo : public codec_softout<libbase::vector, dbl> , private safe_bcjr<
      real, dbl> {
private:
   // Shorthand for class hierarchy
   typedef turbo<real, dbl> This;
   typedef codec_softout<libbase::vector, dbl> Base;
   typedef safe_bcjr<real, dbl> BCJR;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<dbl> array2d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   /*! \name User-defined parameters */
   //! Set of interleavers, one per parity sequence (including first set)
   libbase::vector<interleaver<dbl> *> inter;
   fsm *encoder; //!< Encoder object (same for all parity sequences)
   int iter; //!< Number of iterations to perform
   bool endatzero; //!< Flag to indicate that trellises are terminated
   bool parallel; //!< Flag to enable parallel decoding (rather than serial)
   bool circular; //!< Flag to indicate trellis tailbiting
   // @}
   /*! \name Internal object representation */
   bool initialised; //!< Flag to indicate when memory is initialised
   array2d_t rp; //!< A priori intrinsic source statistics (natural)
   libbase::vector<array2d_t> R; //!< A priori intrinsic encoder-output statistics (interleaved)
   libbase::vector<array2d_t> ra; //!< A priori extrinsic source statistics
   libbase::vector<array1d_t> ss; //!< Holder for start-state probabilities (used with circular trellises)
   libbase::vector<array1d_t> se; //!< Holder for end-state probabilities (used with circular trellises)
   // @}
   /*! \name Internal functions */
   //! Memory allocator (for internal use only)
   void allocate();
   // wrapping functions
   static void work_extrinsic(const array2d_t& ra, const array2d_t& ri,
         const array2d_t& r, array2d_t& re);
   void bcjr_wrap(const int set, const array2d_t& ra, array2d_t& ri,
         array2d_t& re);
   void decode_serial(array2d_t& ri);
   void decode_parallel(array2d_t& ri);
   // @}
protected:
   /*! \name Internal functions */
   void init();
   void free();
   void reset();
   // @}
   /*! \name Codec information functions - internal */
   //! Number of parallel concatenations
   int num_sets() const
      {
      return inter.size();
      }
   //! Size of the interleavers (includes input + tail)
   int inter_size() const
      {
      assertalways(inter.size() > 0);
      assertalways(inter(0));
      return inter(0)->size();
      }
   //! Encoder alphabet size
   int enc_symbols() const
      {
      assert(encoder);
      return encoder->num_symbols();
      }
   //! Number of encoder input symbols / timestep
   int enc_inputs() const
      {
      assert(encoder);
      return encoder->num_inputs();
      }
   //! Number of encoder output symbols / timestep
   int enc_outputs() const
      {
      assert(encoder);
      return encoder->num_outputs();
      }
   //! Number of encoder parity symbols / timestep
   int enc_parity() const
      {
      return enc_outputs() - enc_inputs();
      }
   //! Number of encoder timesteps per block
   int num_timesteps() const
      {
      const int N = inter_size();
      assert(encoder);
      const int k = enc_inputs();
      const int tau = N / k;
      assert(N == tau * k);
      return tau;
      }
   //! Number of encoder states
   int enc_states() const
      {
      assert(encoder);
      return encoder->num_states();
      }
   //! Input alphabet size for algorithm
   int alg_input_symbols() const
      {
      const int k = enc_inputs();
      const int S = enc_symbols();
      return int(pow(S, k));
      }
   //! Output alphabet size for algorithm
   int alg_output_symbols() const
      {
      const int n = enc_outputs();
      const int S = enc_symbols();
      return int(pow(S, n));
      }
   // @}
   // Interface with derived classes
   void advance() const
      {
      // Advance interleavers to the next block
      for (int set = 0; set < num_sets(); set++)
         inter(set)->advance();
      }
   void do_encode(const array1i_t& source, array1i_t& encoded);
   void do_init_decoder(const array1vd_t& ptable);
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app);
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   turbo();
   turbo(const fsm& encoder, const libbase::vector<interleaver<dbl> *>& inter,
         const int iter, const bool endatzero, const bool parallel = false,
         const bool circular = false);
   ~turbo()
      {
      free();
      }
   // @}

   // Codec operations
   void seedfrom(libbase::random& r);
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);
   // (necessary because inheriting methods from templated base)
   using Base::decode;

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> input_block_size() const
      {
      const int tau = num_timesteps();
      const int nu = This::tail_length();
      const int k = enc_inputs();
      return libbase::size_type<libbase::vector>(k * (tau - nu));
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      const int tau = num_timesteps();
      const int k = enc_inputs();
      const int p = enc_parity();
      const int sets = num_sets();
      return libbase::size_type<libbase::vector>(tau * (k + p * sets));
      }
   int num_inputs() const
      {
      return enc_symbols();
      }
   int num_outputs() const
      {
      return enc_symbols();
      }
   int tail_length() const
      {
      assert(encoder);
      return endatzero ? encoder->mem_order() : 0;
      }
   int num_iter() const
      {
      return iter;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(turbo)
};

} // end namespace

#endif
