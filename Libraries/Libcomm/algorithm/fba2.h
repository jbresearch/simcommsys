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
 * \brief   Symbol-Level Forward-Backward Algorithm.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements the forward-backward algorithm for a HMM, as required for the
 * new decoder for Davey & McKay's inner codes, originally introduced in
 * "Watermark Codes: Reliable communication over Insertion/Deletion channels",
 * Trans. IT, 47(2), Feb 2001.
 */

template <class receiver_t, class sig, class real>
class fba2 {
private:
   // Shorthand for class hierarchy
   typedef fba2<receiver_t, sig, real> This;
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
      array4r_t global; // indices (d,i,x,deltax)
      array3r_t local; // indices (d,x,deltax)
   } gamma; //!< Receiver metric
   mutable struct {
      array2b_t global; // indices (i,x)
      array1b_t local; // indices (x)
   } cached; //!< Flag for caching of receiver metric
   array1s_t r; //!< Copy of received sequence, for lazy computation of gamma
   array1vd_t app; //!< Copy of a-priori statistics, for lazy computation of gamma
   int dmin; //!< Offset for deltax index in gamma matrix
   int dmax; //!< Maximum value for deltax index in gamma matrix
   bool initialised; //!< Flag to indicate when memory is allocated
#ifndef NDEBUG
   mutable int gamma_calls; //!< Number of gamma computations
   mutable int gamma_misses; //!< Number of gamma computations causing a cache miss
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
      real result = receiver.R(d, i, r.extract(start, length));
      // apply priors if applicable
      if (app.size() > 0)
         result *= real(app(i)(d));
      return result;
      }
   //! Compute gamma metric using batch receiver interface
   void compute_gamma_batch(int d, int i, int x, array1r_t& ptable,
         const array1s_t& r, const array1vd_t& app) const
      {
      // determine received segment to extract
      const int start = xmax + n * i + x;
      const int length = std::min(n + dmax, r.size() - start);
      // call batch receiver method
      receiver.R(d, i, r.extract(start, length), ptable);
      // apply priors if applicable
      if (app.size() > 0)
         ptable *= real(app(i)(d));
      }
   //! Get a reference to the corresponding gamma cache entry
   real& get_cache_entry(int d, int i, int x, int deltax) const
      {
      if (flags.globalstore)
         return gamma.global[d][i][x][deltax];
      else
         return gamma.local[d][x][deltax];
      }
   //! Fill indicated cache entries for gamma metric - batch interface
   void fill_gamma_cache_batch(int i, int x) const
      {
      // allocate space for results
      static array1r_t ptable;
      ptable.init(2 * dxmax + 1);
      // for each symbol value
      for (int d = 0; d < q; d++)
         {
         // compute metric with batch interface
         compute_gamma_batch(d, i, x, ptable, r, app);
         // store in corresponding place in cache
         for (int deltax = dmin; deltax <= dmax; deltax++)
            get_cache_entry(d, i, x, deltax) = ptable(dxmax + deltax);
         }
      }
   //! Fill indicated cache entries for gamma metric - independent interface
   void fill_gamma_cache_single(int i, int x) const
      {
      // TODO: no need to compute for all deltax if 'cached' is sufficiently fine
      for (int d = 0; d < q; d++)
         {
         const int deltaxmin = std::max(-xmax - x, dmin);
         const int deltaxmax = std::min(xmax - x, dmax);
         for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
            get_cache_entry(d, i, x, deltax) = compute_gamma_single(d, i, x,
                  deltax, r, app);
         }
      }
   //! Wrapper to fill indicated cache entries for gamma metric as needed
   void fill_gamma_cache_conditional(int i, int x) const
      {
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
         // keep track of position index, to reset local cache on change
         static int last_i = -1;
         if (last_i != i)
            {
            last_i = i;
            gamma.local = real(0);
            cached.local = false;
            }
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
            fill_gamma_cache_batch(i, x);
         else
            fill_gamma_cache_single(i, x);
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
#ifndef NDEBUG
      gamma_calls++;
#endif
      // pre-computed values
      if (!flags.lazy)
         return gamma.global[d][i][x][deltax];
      // lazy computation, with local or global storage
      fill_gamma_cache_conditional(i, x);
      return get_cache_entry(d, i, x, deltax);
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
   // specialized components for decode funtions
   void work_gamma_single(const array1s_t& r, const array1vd_t& app);
   void work_gamma_batch(const array1s_t& r, const array1vd_t& app);
   void work_message_app(array1vr_t& ptable) const;
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
   // decode functions
   void work_gamma(const array1s_t& r, const array1vd_t& app);
   void work_alpha(const array1d_t& sof_prior);
   void work_beta(const array1d_t& eof_prior);
   void work_results(array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2()
      {
      initialised = false;
      }
   // @}

   // main initialization routine - constructor essentially just calls this
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner,
         double th_outer, bool norm, bool batch, bool lazy, bool globalstore);

   /*! \name Parameter getters */
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
   void get_drift_pdf(array1vr_t& pdftable) const;

   // Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm";
      }
};

} // end namespace

#endif
