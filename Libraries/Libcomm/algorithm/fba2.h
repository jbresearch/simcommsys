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

#ifndef __fba2_h
#define __fba2_h

#include "fba2-interface.h"
#include "matrix.h"
#include "multi_array.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
 * \brief   Symbol-Level Forward-Backward Algorithm (for TVB codes).
 * \author  Johann Briffa
 *
 * Implements the forward-backward algorithm for a HMM, as required for the
 * MAP decoding algorithm for a generalized class of synchronization-correcting
 * codes described in
 * Johann A. Briffa, Victor Buttigieg, and Stephan Wesemeyer, "Time-varying
 * block codes for synchronisation errors: maximum a posteriori decoder and
 * practical issues. IET Journal of Engineering, 30 Jun 2014.
 *
 * \tparam receiver_t Type for receiver metric computer
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 * \tparam thresholding Flag to indicate if we're doing path thresholding
 * \tparam lazy Flag indicating lazy computation of gamma metric
 * \tparam globalstore Flag indicating global pre-computation or caching of gamma values
 */

template <class receiver_t, class sig, class real, class real2,
      bool thresholding, bool lazy, bool globalstore>
class fba2 : public fba2_interface<sig, real, real2> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   typedef boost::assignable_multi_array<real, 2> array2r_t;
   typedef boost::assignable_multi_array<real, 3> array3r_t;
   typedef boost::assignable_multi_array<real, 4> array4r_t;
   typedef boost::assignable_multi_array<bool, 1> array1b_t;
   typedef boost::assignable_multi_array<bool, 2> array2b_t;
   // @}
private:
   /*! \name Internally-used objects */
   mutable receiver_t receiver; //!< Inner code receiver metric computation
   array2r_t alpha; //!< Forward recursion metric
   array2r_t beta; //!< Backward recursion metric
   mutable struct {
      array4r_t global; // indices (i,x,d,deltax)
      array3r_t local; // indices (x,d,deltax)
   } gamma; //!< Receiver metric
   mutable struct {
      array2b_t global; // indices (i,x)
      array1b_t local; // indices (x)
   } cached; //!< Flag for caching of receiver metric
   mutable array1i_t cw_length; //!< Codeword 'i' length
   mutable array1i_t cw_start; //!< Codeword 'i' start
   mutable int tau; //!< Frame length (all codewords in sequence)
   array1s_t r; //!< Copy of received sequence, for lazy or local computation of gamma
   array1vd_t app; //!< Copy of a-priori statistics, for lazy or local computation of gamma
   bool initialised; //!< Flag to indicate when memory is allocated
#ifndef NDEBUG
   mutable int gamma_calls; //!< Number of calls requesting gamma values
   mutable int gamma_misses; //!< Number of cache misses in such calls
#endif
   // @}
   /*! \name User-defined parameters */
   real th_inner; //!< Threshold factor for inner cycle
   real th_outer; //!< Threshold factor for outer cycle
   int N; //!< The transmitted block size in symbols
   int q; //!< The number of symbols in the q-ary alphabet
   int mtau_min; //!< The largest negative drift within a whole frame is \f$ m_\tau^{-} \f$
   int mtau_max; //!< The largest positive drift within a whole frame is \f$ m_\tau^{+} \f$
   int mn_min; //!< The largest negative drift within a q-ary symbol is \f$ m_n^{-} \f$
   int mn_max; //!< The largest positive drift within a q-ary symbol is \f$ m_n^{+} \f$
   int m1_min; //!< The largest negative drift over a single channel symbol is \f$ m_1^{-} \f$
   int m1_max; //!< The largest positive drift over a single channel symbol is \f$ m_1^{+} \f$
   // @}
private:
   /*! \name Internal functions - computer */
   //! Get a reference to the corresponding gamma storage entry
   real& gamma_storage_entry(int d, int i, int x, int deltax) const
      {
      if (globalstore)
         return gamma.global[i][x][d][deltax];
      else
         return gamma.local[x][d][deltax];
      }
   //! Fill indicated storage entries for gamma metric - batch interface
   void fill_gamma_storage_batch(const array1s_t& r, const array1vd_t& app, int i, int x) const
      {
      // allocate space for results
      static array1r_t ptable;
      ptable.init(mn_max - mn_min + 1);
      // determine received segment to extract
      // n * i = offset to start of current codeword
      // -mtau_min = offset to zero drift in 'r'
      const int start = cw_start(i) + x - mtau_min;
      const int length = std::min(cw_length(i) + mn_max, r.size() - start);
      // for each symbol value
      for (int d = 0; d < q; d++)
         {
         // call batch receiver method
         receiver.R(d, i, r.extract(start, length), app, ptable);
         // store in corresponding place in storage
         for (int deltax = mn_min; deltax <= mn_max; deltax++)
            gamma_storage_entry(d, i, x, deltax) = ptable(deltax - mn_min);
         }
      }
   /*! \brief Fill indicated cache entries for gamma metric as needed
    *
    * This method is called on every get_gamma call when doing lazy computation.
    * It will update the cache as needed, for both local/global storage.
    */
   void fill_gamma_cache_conditional(int i, int x) const
      {
#ifndef NDEBUG
      gamma_calls++;
#endif
      bool miss = false;
      if (globalstore)
         {
         // if we not have this already, mark to fill in this part of cache
         if (!cached.global[i][x])
            {
            miss = true;
            cached.global[i][x] = true;
            }
         }
      else
         {
         // if we not have this already, mark to fill in this part of cache
         if (!cached.local[x])
            {
            miss = true;
            cached.local[x] = true;
            }
         }
      if (miss)
         {
#ifndef NDEBUG
         gamma_misses++;
#endif
         fill_gamma_storage_batch(r, app, i, x);
         }
      }
   /*! \brief Wrapper for retrieving gamma metric value
    * This method is called from nested loops as follows:
    * - from work_alpha:
    *        i,x,d=outer loops
    *        deltax=inner loop
    * - from work_beta:
    *        i,x,d=outer loops
    *        deltax=inner loop
    * - from work_message_app:
    *        i,d,x=outer loops
    *        deltax=inner loop
    */
   real get_gamma(int d, int i, int x, int deltax) const
      {
      // update cache values if necessary
      if (lazy)
         fill_gamma_cache_conditional(i, x);
      return gamma_storage_entry(d, i, x, deltax);
      }
   // common small tasks
   static real get_threshold(const array2r_t& metric, int row, int col_min,
         int col_max, real factor);
   static real get_scale(const array2r_t& metric, int row, int col_min,
         int col_max);
   static void normalize(array2r_t& metric, int row, int col_min, int col_max);
   void normalize_alpha(int i)
      {
      normalize(alpha, i, mtau_min, mtau_max);
      }
   void normalize_beta(int i)
      {
      normalize(beta, i, mtau_min, mtau_max);
      }
   // decode functions - partial computations
   void work_gamma(const array1s_t& r, const array1vd_t& app,
         const int i) const
      {
      for (int x = mtau_min; x <= mtau_max; x++)
         fill_gamma_storage_batch(r, app, i, x);
      }
   void work_alpha(const int i);
   void work_beta(const int i);
   void work_message_app(array1vr_t& ptable, const int i) const;
   void work_state_app(array1r_t& ptable, const int i) const;
   // @}
private:
   /*! \name Internal functions - main */
   // memory allocation
   void allocate();
   void free();
   // helper methods
   void reset_cache() const;
   void print_gamma(std::ostream& sout) const;
   // decode functions - global path
   void work_gamma(const array1s_t& r, const array1vd_t& app);
   void work_alpha_and_beta(const array1d_t& sof_prior,
         const array1d_t& eof_prior);
   void work_results(array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post) const;
   // decode functions - local path
   void work_alpha(const array1d_t& sof_prior);
   void work_beta_and_results(const array1d_t& eof_prior, array1vr_t& ptable,
         array1r_t& sof_post, array1r_t& eof_post);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2() :
         initialised(false)
      {
      }
   // @}

   /*! \name FBA2 Interface Implementation */
   /*! \brief Set up code size, decoding parameters, and channel receiver
    * Only needs to be done before the first frame.
    */
   void init(int N, int q, int mtau_min, int mtau_max, int mn_min, int mn_max,
         int m1_min, int m1_max, double th_inner, double th_outer,
         const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer);
   /*! \brief Set up encoding table
    * Needs to be done before every frame.
    */
   void init(const array2vs_t& encoding_table) const
      {
      // Initialize arrays with start and length of each codeword
      cw_length.init(N);
      cw_start.init(N);
      int start = 0;
      for (int i = 0; i < N; i++)
         {
         const int n = encoding_table(i, 0).size();
         cw_start(i) = start;
         cw_length(i) = n;
         start += n;
         }
      tau = start;
      // Set up receiver with new encoding table
      this->receiver.init(encoding_table);
      }

   // decode functions
   void decode(libcomm::instrumented& collector, const array1s_t& r,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post, const int offset);
   void get_drift_pdf(array1r_t& pdf, const int i) const
      {
      work_state_app(pdf, i);
      }
   void get_drift_pdf(array1vr_t& pdftable) const;

   //! Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm";
      }
   // @}
};

} // end namespace

#endif
