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

#ifndef __fba2_h
#define __fba2_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "multi_array.h"
#include "fsm.h"
#include "instrumented.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
 * \brief   Symbol-Level Forward-Backward Algorithm (for TVB codes).
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements the forward-backward algorithm for a HMM, as required for the
 * MAP decoding algorithm for a generalized class of synchronization-correcting
 * codes described in
 * Briffa et al, "A MAP Decoder for a General Class of Synchronization-
 * Correcting Codes", Submitted to Trans. IT, 2011.
 *
 * \tparam receiver_t Type for receiver metric computer
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 */

template <class receiver_t, class sig, class real, class real2>
class fba2 {
private:
   // Shorthand for class hierarchy
   typedef fba2<receiver_t, sig, real, real2> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<int> array1i_t;
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
   array1s_t r; //!< Copy of received sequence, for lazy or local computation of gamma
   array1vd_t app; //!< Copy of a-priori statistics, for lazy or local computation of gamma
   int dmin; //!< Offset for deltax index in gamma matrix
   int dmax; //!< Maximum value for deltax index in gamma matrix
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
   int n; //!< The number of bits encoding each q-ary symbol
   int q; //!< The number of symbols in the q-ary alphabet
   int I; //!< The maximum number of insertions considered before every transmission
   int xmax; //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
   int dxmax; //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
   struct {
      bool norm; //!< Flag to indicate if metrics should be normalized between time-steps
      bool batch; //!< Flag indicating use of batch receiver interface
      bool lazy; //!< Flag indicating lazy computation of gamma metric
      bool globalstore; //!< Flag indicating global pre-computation or caching of gamma values
   } flags;
   // @}
private:
   /*! \name Internal functions - computer */
   //! Compute gamma metric using independent receiver interface
   real compute_gamma_single(int d, int i, int x, int deltax,
         const array1s_t& r, const array1vd_t& app) const
      {
      // determine received segment to extract
      const int start = xmax + n * i + x;
      const int length = n + deltax;
      // call receiver method
      return receiver.R(d, i, r.extract(start, length), app);
      }
   //! Compute gamma metric using batch receiver interface
   void compute_gamma_batch(int d, int i, int x, array1r_t& ptable,
         const array1s_t& r, const array1vd_t& app) const
      {
      // determine received segment to extract
      const int start = xmax + n * i + x;
      const int length = std::min(n + dmax, r.size() - start);
      // call batch receiver method
      receiver.R(d, i, r.extract(start, length), app, ptable);
      }
   //! Get a reference to the corresponding gamma storage entry
   real& gamma_storage_entry(int d, int i, int x, int deltax) const
      {
      if (flags.globalstore)
         return gamma.global[i][x][d][deltax];
      else
         return gamma.local[x][d][deltax];
      }
   //! Fill indicated storage entries for gamma metric - batch interface
   void fill_gamma_storage_batch(const array1s_t& r, const array1vd_t& app, int i, int x) const
      {
      // allocate space for results
      static array1r_t ptable;
      ptable.init(2 * dxmax + 1);
      // for each symbol value
      for (int d = 0; d < q; d++)
         {
         // compute metric with batch interface
         compute_gamma_batch(d, i, x, ptable, r, app);
         // store in corresponding place in storage
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma_storage_entry(d, i, x, deltax) = ptable(dxmax + deltax);
         }
      }
   /*! \brief Fill indicated storage entries for gamma metric - independent interface
    * \todo No need to compute for all deltax if 'cached' is sufficiently fine
    */
   void fill_gamma_storage_single(const array1s_t& r, const array1vd_t& app, int i, int x) const
      {
      // limit on end-state (-xmax <= x2 <= xmax):
      //   x2-x1 <= xmax-x1
      //   x2-x1 >= -xmax-x1
      const int deltaxmin = std::max(-xmax - x, dmin);
      const int deltaxmax = std::min(xmax - x, dmax);
      for (int d = 0; d < q; d++)
         {
         // clear gamma entries
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma_storage_entry(d, i, x, deltax) = 0;
         // compute entries within required limits
         for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
            gamma_storage_entry(d, i, x, deltax) = compute_gamma_single(d, i, x,
                  deltax, r, app);
         }
      }
   /*! \brief Fill indicated cache entries for gamma metric as needed
    *
    * This method is called on every get_gamma call when doing lazy computation.
    * It will update the cache as needed, for both local/global storage,
    * and choosing between batch/single methods as required.
    */
   void fill_gamma_cache_conditional(int i, int x) const
      {
#ifndef NDEBUG
      gamma_calls++;
#endif
      bool miss = false;
      if (flags.globalstore)
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
         // call computation method and store results
         if (flags.batch)
            fill_gamma_storage_batch(r, app, i, x);
         else
            fill_gamma_storage_single(r, app, i, x);
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
      if (flags.lazy)
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
      normalize(alpha, i, -xmax, xmax);
      }
   void normalize_beta(int i)
      {
      normalize(beta, i, -xmax, xmax);
      }
   // decode functions - partial computations
   void work_gamma(const array1s_t& r, const array1vd_t& app,
         const int i) const
      {
      for (int x = -xmax; x <= xmax; x++)
         if (flags.batch)
            fill_gamma_storage_batch(r, app, i, x);
         else
            fill_gamma_storage_single(r, app, i, x);
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
   /*! \brief Copy constructor
    * \note Copy construction is a deep copy, except for:
    *       - metric tables (alpha, beta, gamma, cached)
    *
    * \todo This will not be necessary (can keep the default copy constructor)
    *       when/if the TX and RX side of commsys objects are separated, as we
    *       won't need to clone the RX commsys object in stream simulations.
    */
   fba2(const fba2<receiver_t, sig, real, real2>& x) :
         receiver(x.receiver), r(x.r), app(x.app), dmin(x.dmin), dmax(x.dmax),
         initialised(false), th_inner(x.th_inner), th_outer(x.th_outer),
         N(x.N), n(x.n), q(x.q), I(x.I), xmax(x.xmax), dxmax(x.dxmax),
         flags(x.flags)
      {
      }
   // @}

   // main initialization routine - constructor essentially just calls this
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner,
         double th_outer, bool norm, bool batch, bool lazy, bool globalstore);

   /*! \name Parameter getters */
   //! Determine memory required for global storage mode (in MiB)
   static int get_memory_required(int N, int n, int q, int I, int xmax,
         int dxmax)
      {
      // determine allowed limits on deltax: see allocate() for documentation
      const int dmin = std::max(-n, -dxmax);
      const int dmax = std::min(n * I, dxmax);
      // determine memory required
      // NOTE: do all computations at 64-bit, or we get intermediate overflow!
      libbase::int64u bytes_required = sizeof(real);
      bytes_required *= q;
      bytes_required *= N;
      bytes_required *= (2 * xmax + 1);
      bytes_required *= (dmax - dmin + 1);
      bytes_required >>= 20;
      return int(bytes_required);
      }
   //! Access metric computation
   receiver_t& get_receiver() const
      {
      return receiver;
      }
   int get_N() const
      {
      return N;
      }
   int get_n() const
      {
      return n;
      }
   int get_q() const
      {
      return q;
      }
   int get_I() const
      {
      return I;
      }
   int get_xmax() const
      {
      return xmax;
      }
   int get_dxmax() const
      {
      return dxmax;
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

   // Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm";
      }
};

} // end namespace

#endif
