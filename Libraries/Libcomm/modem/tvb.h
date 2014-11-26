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

#ifndef __tvb_h
#define __tvb_h

#include "config.h"

#include "stream_modulator.h"
#include "channel_insdel.h"
#include "algorithm/fba2-interface.h"

#include "randgen.h"
#include "itfunc.h"
#include "vector_itfunc.h"
#include "field_utils.h"
#include "channel/qids-utils.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>
#include <memory>

#include "boost/shared_ptr.hpp"

#ifdef USE_CUDA
#  include "tvb-receiver-cuda.h"
#else
#  include "tvb-receiver.h"
#endif

namespace libcomm {

/*!
 * \brief   Time-Varying Block Code.
 * \author  Johann Briffa
 *
 * Implements a MAP decoding algorithm for a generalized class of
 * synchronization-correcting codes. The algorithm is described in
 * Briffa et al, "A MAP Decoder for a General Class of Synchronization-
 * Correcting Codes", Submitted to Trans. IT, 2011.
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 */

template <class sig, class real, class real2>
class tvb : public stream_modulator<sig> , public parametric {
private:
   // Shorthand for class hierarchy
   typedef stream_modulator<sig> Interface;
   typedef tvb<sig, real, real2> This;
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
   enum codebook_t {
      codebook_sparse = 0, //!< as in DM construction
      codebook_random, //!< randomly-constructed codebooks, update every frame
      codebook_user_sequential, //!< sequentially-applied user codebooks
      codebook_user_random, //!< randomly-applied user codebooks
      codebook_undefined
   };
   enum marker_t {
      marker_zero = 0, //!< no marker sequence
      marker_random, //!< random marker sequence, update every frame
      marker_user_sequential, //!< sequentially-applied user sequence
      marker_user_random, //!< randomly-applied user sequence
      marker_undefined
   };
   enum storage_t {
      storage_local = 0, //!< always use local storage
      storage_global, //!< always use global storage
      storage_conditional, //!< use global storage below memory limit
      storage_undefined
   };
   // @}
private:
   /*! \name User-defined parameters */
   int n; //!< codeword length in symbols
   int q; //!< number of codewords (input alphabet size)
   marker_t marker_type; //!< enum indicating marker sequence type
   array1vs_t marker_vectors; //!< user set of marker vectors
   codebook_t codebook_type; //!< enum indicating codebook type
   std::string codebook_name; //!< name to describe codebook
   array2vs_t codebook_tables; //!< user set of codebooks
   real th_inner; //!< Threshold factor for inner cycle
   real th_outer; //!< Threshold factor for outer cycle
   double Pr; //!< Probability of channel event outside chosen limits
   struct {
      bool lazy; //!< Flag indicating lazy computation of gamma metric
   } flags;
   storage_t storage_type; //!< enum indicating storage mode for gamma metric
   int globalstore_limit; //!< fba memory threshold in MiB for global storage, if applicable
   int lookahead; //!< Number of codewords to look ahead when stream decoding
   // @}
   /*! \name Internally-used objects */
   std::auto_ptr<channel_insdel<sig,real2> > mychan; //!< bound channel object
   mutable libbase::randgen r; //!< for construction and random application of codebooks and marker sequence
   mutable array2vs_t encoding_table; //!< per-frame encoding table
   mutable bool changed_encoding_table; //!< flag indicating encoding table has changed since last use
   int mtau_min; //!< The largest negative drift within a whole frame is \f$ m_\tau^{-} \f$
   int mtau_max; //!< The largest positive drift within a whole frame is \f$ m_\tau^{+} \f$
#ifdef USE_CUDA
   typedef cuda::tvb_receiver<sig, real, real2> recv_type;
#else
   typedef tvb_receiver<sig, real, real2> recv_type;
#endif
   typedef fba2_interface<sig, real, real2> fba_type;
   boost::shared_ptr<fba_type> fba_ptr; //!< pointer to algorithm object
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
   int select_codebook(const int i) const;
   int select_marker(const int i) const;
   void fill_encoding_table(array2vs_t& encoding_table, const int offset,
         const int length) const;
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
   //! Invariance test
   void test_invariant() const
      {
      // check code parameters
      assert(n >= 1);
      assert(q >= 2 && q <= int(pow(field_utils<sig>::elements(), n)));
      // check cutoff thresholds
      assert(th_inner >= real(0) && th_inner <= real(1));
      assert(th_outer >= real(0) && th_outer <= real(1));
      }
   // codebook wrapper operations
   void validate_sequence_length(const array1vs_t& table) const;
   void copycodebook(const int i, const array1vs_t& codebook_s);
   void showcodebook(std::ostream& sout, const array1vs_t& codebook) const;
   void showcodebooks(std::ostream& sout) const;
   void validatecodebook() const;
   // Other utilities
   void checkforchanges(int m1_min, int m1_max, int mn_min,
         int mn_max, int mtau_min, int mtau_max) const;
   void checkforchanges(bool globalstore, int required) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   explicit tvb(const int n = 2, const int q = 2, const double th_inner = 0,
         const double th_outer = 0) :
      n(n), q(q), marker_type(marker_random), codebook_type(codebook_random),
            th_inner(real(th_inner)), th_outer(real(th_outer))
      {
      init();
      }
   // @}
   /*! \name Fix to avoid duplicate memory usage. */
   /*! \brief Copy constructor
    * Copies all serialized elements and internal state but excluding
    * the FBA object; this avoids having a duplicate copy of its tables.
    *
    * \todo This will not be necessary (can keep the compiler defaults)
    *       when/if the TX and RX side of commsys objects are separated, as we
    *       won't need to clone the RX commsys object in stream simulations.
    */
   tvb(const tvb& x) :
         n(x.n), q(x.q), marker_type(x.marker_type), marker_vectors(
               x.marker_vectors), codebook_type(x.codebook_type), codebook_name(
               x.codebook_name), codebook_tables(x.codebook_tables), th_inner(
               x.th_inner), th_outer(x.th_outer), Pr(x.Pr), flags(x.flags), storage_type(
               x.storage_type), globalstore_limit(x.globalstore_limit), lookahead(
               x.lookahead), mychan(
               dynamic_cast<channel_insdel<sig, real2>*>(x.mychan->clone())), r(x.r), encoding_table(
               x.encoding_table), changed_encoding_table(
               x.changed_encoding_table)
      {
      }
   // @}

   /*! \name Marker-specific setup functions */
   void set_thresholds(const real th_inner, const real th_outer);
   void set_parameter(const double x)
      {
      set_thresholds(real(x), real(x));
      }
   double get_parameter() const
      {
      assert(th_inner == th_outer);
      return th_inner;
      }
   // @}

   /*! \name TVB-specific informative functions */
   int get_symbolsize() const
      {
      return n;
      }
   int num_codebooks() const
      {
      return codebook_tables.size().rows();
      }
   array1s_t get_symbol(int i, int d) const
      {
      assert(i >= 0 && i < num_codebooks());
      return codebook_tables(i, d);
      }
   double get_th_inner() const
      {
      return th_inner;
      }
   double get_th_outer() const
      {
      return th_outer;
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
      return q;
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(this->input_block_size() * n);
      }
   double energy() const
      {
      return n;
      }

   // Block modem operations - streaming extensions
   void get_post_drift_pdf(array1vd_t& pdftable) const
      {
      // Inherit sizes
      const int N = this->input_block_size();
      // get the posterior channel drift pdf at codeword boundaries
      array1vr_t pdftable_r;
      fba_ptr->get_drift_pdf(pdftable_r);
      libbase::normalize_results(pdftable_r.extract(0, N + 1), pdftable);
      }
   array1i_t get_boundaries(void) const
      {
      // Inherit sizes
      const int N = this->input_block_size();
      // construct list of codeword boundary positions
      array1i_t postable(N + 1);
      for (int i = 0; i <= N; i++)
         postable(i) = i * n;
      return postable;
      }
   libbase::size_type<libbase::vector> get_suggested_lookahead(void) const
      {
      return libbase::size_type<libbase::vector>(n * lookahead);
      }
   double get_suggested_exclusion(void) const
      {
      return Pr;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(tvb)
};

} // end namespace

#endif
