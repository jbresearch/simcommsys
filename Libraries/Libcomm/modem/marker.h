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

#ifndef __marker_h
#define __marker_h

#include "config.h"

#include "stream_modulator.h"
#include "channel/qids.h"

#include "randgen.h"
#include "itfunc.h"
#include "vector_itfunc.h"
#include "field_utils.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>
#include <memory>

#include "algorithm/fba_generic.h"

namespace libcomm {

/*!
 * \brief   Marker Code.
 * \author  Johann Briffa
 *
 * Implements a bit-level MAP decoding algorithm for data protected by marker
 * sequences. The original construction and decoding algorithm are described in
 * Ratzer, "Error-correction on non-standard communication channels",
 * PhD dissertation, University of Cambridge, 2003.
 * This class implements decoding using a trellis-based implementation of the
 * forward-backward algorithm, rather than a lattice-based one.
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 */

template <class sig, class real, class real2>
class marker : public stream_modulator<sig> {
private:
   // Shorthand for class hierarchy
   typedef marker<sig, real, real2> This;
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
   enum marker_t {
      marker_random = 0, //!< random marker sequence
      marker_user_sequential, //!< sequentially-applied user sequence
      marker_user_random, //!< randomly-applied user sequence
      marker_undefined
   };
   // @}
private:
   /*! \name User-defined parameters */
   int d; //!< number of data symbols between markers
   int m; //!< length of marker sequence
   marker_t marker_type; //!< enum indicating marker sequence type
   array1vs_t marker_vectors; //!< user set of marker vectors
   double Pr; //!< Probability of channel event outside chosen limits
   bool norm; //!< Flag to indicate if metrics should be normalized between time-steps
   int lookahead; //!< Number of codewords to look ahead when stream decoding
   // @}
   /*! \name Internally-used objects */
   qids<sig, real2> mychan; //!< bound channel object
   mutable libbase::randgen r; //!< for construction and random application of markers
   mutable array1vs_t frame_marker_sequence; //!< per-frame sequence of markers
   fba_generic<sig, real, real2> fba; //!< algorithm object
   // @}
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
   // Internal methods
   array1s_t select_marker(const int i) const;
   void fill_frame_marker_sequence(array1vs_t& frame_marker_sequence,
         const int offset, const int length) const;
   void demodulate_wrapper(const channel<sig>& chan, const array1s_t& rx,
         const int lookahead, const array1d_t& sof_prior,
         const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
         array1d_t& sof_post, array1d_t& eof_post, const int offset);
private:
   /*! \name Internal functions */
   // Setup function
   void init(const channel<sig>& chan, const array1d_t& sof_pdf,
         const int offset);
   void init(const channel<sig>& chan)
      {
      const array1d_t eof_pdf;
      const int offset = 0;
      init(chan, eof_pdf, offset);
      }
   void init();
   // Invariance test
   void test_invariant() const
      {
      // check code parameters
      assert(d >= 1);
      assert(m >= 0);
      }
   // codebook wrapper operations
   void validate_marker_length(const array1vs_t& table) const;
   // Other utilities
   void checkforchanges(int m1_min, int m1_max, int mtau_min, int mtau_max) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   explicit marker(const int d = 1, const int m = 0) :
         d(d), m(m), marker_type(marker_random)
      {
      init();
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
      return field_utils<sig>::elements();
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      // determine number of codewords (data + marker sequences)
      const int N = this->input_block_size() / d;
      assertalways(this->input_block_size() == N * d);
      return libbase::size_type<libbase::vector>(N * (d + m));
      }
   double energy() const
      {
      return (d + m) / double(d);
      }

   // Block modem operations - streaming extensions
   void get_post_drift_pdf(array1vd_t& pdftable,
         libbase::size_type<libbase::vector>& offset) const
      {
      // Inherit sizes
      const int N = this->input_block_size() / d;
      // get the posterior channel drift pdf at codeword boundaries
      array1r_t pdf_r;
      pdftable.init(N + 1);
      for (int i = 0; i <= N; i++)
         {
         fba.get_drift_pdf(pdf_r, i * (d + m));
         libbase::normalize(pdf_r, pdftable(i));
         }
      // set the offset used
      offset = libbase::size_type<libbase::vector>(-fba.get_mtau_min());
      }
   array1i_t get_boundaries(void) const
      {
      // Inherit sizes
      const int N = this->input_block_size() / d;
      // construct list of codeword boundary positions
      array1i_t postable(N + 1);
      for (int i = 0; i <= N; i++)
         postable(i) = i * (d + m);
      return postable;
      }
   libbase::size_type<libbase::vector> get_suggested_lookahead(void) const
      {
      return libbase::size_type<libbase::vector>((d + m) * lookahead);
      }
   double get_suggested_exclusion(void) const
      {
      return Pr;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(marker)
};

} // end namespace

#endif
