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

#ifndef __dminner_h
#define __dminner_h

#include "config.h"

#include "stream_modulator.h"
#include "algorithm/fba.h"
#include "channel/bsid.h"

#include "bitfield.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>
#include <memory>

namespace libcomm {

/*!
 * \brief   Davey-MacKay Inner Code, original bit-level decoding.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements 'Watermark' Codes as described by Davey and MacKay in "Reliable
 * Communication over Channels with Insertions, Deletions, and Substitutions",
 * Trans. IT, Feb 2001.
 *
 * \note In demodulate(), the ptable is internally computed as type 'real',
 * and then copied over after normalization. We norm over the whole
 * block instead of independently for each timestep. This should be
 * equivalent to no-normalization, and is a precursor to a change in the
 * architecture to allow higher-range ptables.
 *
 * \todo Separate this class from friendship with dminner2; common elements
 * should be extracted into a common base
 */

template <class real>
class dminner2;

template <class real>
class dminner : public stream_modulator<bool> , public parametric, private fba<
      bool, real> {
   friend class dminner2<real> ;
private:
   // Shorthand for class hierarchy
   typedef dminner<real> This;
   typedef fba<bool, real> FBA;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   enum codebook_t {
      codebook_sparse = 0, codebook_user, codebook_tvb
   };
   enum marker_t {
      marker_random = 0, marker_zero, marker_alt_symbol, marker_mod_vec
   };
   // @}
private:
   /*! \name User-defined parameters */
   int n; //!< number of bits in sparse (output) symbol
   int k; //!< number of bits in message (input) symbol
   marker_t marker_type; //!< enum indicating codebook type
   array1i_t marker_vectors; //!< modification vectors
   codebook_t codebook_type; //!< enum indicating codebook type
   std::string codebookname; //!< name to describe codebook
   array2i_t codebook; //!< codebook
   bool user_threshold; //!< flag indicating that thresholds are supplied by user
   real th_inner; //!< Threshold factor for inner cycle
   real th_outer; //!< Threshold factor for outer cycle
   bool norm; //!< Flag to indicate if metrics should be normalized between time-steps
   // @}
   /*! \name Pre-computed parameters */
   double f; //!< average weight per codeword bit
   // @}
   /*! \name Internally-used objects */
   bsid mychan; //!< bound channel object
   mutable libbase::randgen r; //!< marker sequence generator
   mutable array1i_t marker; //!< marker sequence
   // @}
private:
   // Implementations of channel-specific metrics for fba
   real R(const int i, const array1b_t& r);
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const
      {
      failwith("Function should not be used.");
      return false;
      }
   const int demodulate(const bool& signal) const
      {
      failwith("Function should not be used.");
      return 0;
      }
   const int demodulate(const bool& signal, const libbase::vector<double>& app) const
      {
      failwith("Function should not be used.");
      return 0;
      }
protected:
   // Interface with derived classes
   void advance() const;
   void domodulate(const int N, const array1i_t& encoded, array1b_t& tx);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         array1vd_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         const array1vd_t& app, array1vd_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
         array1d_t& eof_post, const libbase::size_type<libbase::vector> offset);

private:
   /*! \name Internal functions */
   void test_invariant() const
      {
      // check code parameters
      assert(n >= 1 && n <= 32);
      assert(k >= 1 && k <= n);
      // check cutoff thresholds
      assert(th_inner >= real(0) && th_inner <= real(1));
      assert(th_outer >= real(0) && th_outer <= real(1));
      }
   // codebook wrapper operations
   array1b_t encode(const int i, const int d) const;
   void validate_bitfield_length(
         const libbase::vector<libbase::bitfield>& table) const;
   void copymarker(const libbase::vector<libbase::bitfield>& marker_b);
   void copycodebook(const int i,
         const libbase::vector<libbase::bitfield>& codebook_b);
   void showcodebook(std::ostream& sout) const;
   void validatecodebook() const;
   void computemeandensity();
   // Other utilities
   void checkforchanges(int I, int xmax) const;
   void work_results(const array1b_t& r, array1vr_t& ptable, const int xmax,
         const int dxmax, const int I) const;
   void normalize_results(const array1vr_t& in, array1vd_t& out) const;
   // @}
protected:
   /*! \name Internal functions */
   void init();
   // @}
public:
   /*! \name Constructors / Destructors */
   explicit dminner(const int n = 2, const int k = 1) :
      n(n), k(k), codebook_type(codebook_sparse), user_threshold(false)
      {
      init();
      }
   dminner(const int n, const int k, const double th_inner,
         const double th_outer) :
      n(n), k(k), codebook_type(codebook_sparse), user_threshold(true),
            th_inner(real(th_inner)), th_outer(real(th_outer))
      {
      init();
      }
   // @}

   /*! \name Marker-specific setup functions */
   void set_marker(libbase::vector<bool> marker);
   void set_marker(libbase::vector<libbase::bitfield> marker);
   void set_codebook(libbase::vector<libbase::bitfield> codebook_b);
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

   /*! \name Watermark-specific informative functions */
   int get_symbolsize() const
      {
      return n;
      }
   int num_codebooks() const
      {
      return codebook.size().rows();
      }
   int get_symbol(int i, int d) const
      {
      assert(i >= 0 && i < num_codebooks());
      return codebook(i, d);
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
      return 1 << k;
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(input_block_size() * n);
      }
   double energy() const
      {
      return n;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dminner)
};

} // end namespace

#endif
