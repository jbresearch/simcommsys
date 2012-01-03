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
#include "modem/dminner2-receiver.h"
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

template <class real, class sig, bool norm>
class fba2 {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   typedef boost::assignable_multi_array<real, 2> array2r_t;
   typedef boost::assignable_multi_array<real, 4> array4r_t;
   typedef boost::assignable_multi_array<bool, 2> array2b_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef fba2<real, sig, norm> This;
private:
   /*! \name User-defined parameters */
   int N; //!< The transmitted block size in symbols
   int n; //!< The number of bits encoding each q-ary symbol
   int q; //!< The number of symbols in the q-ary alphabet
   int I; //!< The maximum number of insertions considered before every transmission
   int xmax; //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
   int dxmax; //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
   real th_inner; //!< Threshold factor for inner cycle
   real th_outer; //!< Threshold factor for outer cycle
   // @}
   /*! \name Internally-used objects */
   int dmin; //!< Offset for deltax index in gamma matrix
   int dmax; //!< Maximum value for deltax index in gamma matrix
   bool initialised; //!< Flag to indicate when memory is allocated
   bool cache_enabled; //!< Flag to indicate when cache is usable
   array2r_t alpha; //!< Forward recursion metric
   array2r_t beta; //!< Backward recursion metric
   mutable array4r_t gamma; //!< Receiver metric
   mutable array2b_t cached; //!< Flag for caching of receiver metric
   array1s_t r; //!< Copy of received sequence, for lazy computation of gamma
   array1vd_t app; //!< Copy of a-priori statistics, for lazy computation of gamma
#ifndef NDEBUG
   mutable int gamma_calls; //!< Number of gamma computations
   mutable int gamma_misses; //!< Number of gamma computations causing a cache miss
#endif
   mutable dminner2_receiver<real> receiver; //!< Inner code receiver metric computation
   // @}
private:
   /*! \name Internal functions */
   void compute_gamma(int d, int i, int x, array1r_t& ptable) const;
   real compute_gamma(int d, int i, int x, int deltax) const;
   real get_gamma(int d, int i, int x, int deltax) const;
   // memory allocation
   void allocate();
   void free();
   void reset_cache() const;
   // @}
protected:
   /*! \name Internal functions */
   // decode functions
   void work_gamma(const array1s_t& r, const array1vd_t& app);
   void work_alpha(const array1d_t& sof_prior);
   void work_beta(const array1d_t& eof_prior);
   void work_message_app(array1vr_t& ptable) const;
   void work_state_app(array1r_t& ptable, const int i) const;
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
         double th_outer);

   /*! \name Parameter getters */
   //! Access metric computation
   dminner2_receiver<real>& get_receiver() const
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

   // Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm";
      }
};

template <class real, class sig, bool norm>
inline void fba2<real, sig, norm>::compute_gamma(int d, int i, int x,
      array1r_t& ptable) const
   {
   // determine received segment to extract
   const int start = xmax + n * i + x;
   const int length = std::min(n + dmax, r.size() - start);
   // set up space for results
   static libbase::vector<bsid::real> ptable_r;
   ptable_r.init(ptable.size());
   // call batch receiver method and convert results
   receiver.R(d, i, r.extract(start, length), ptable_r);
   ptable = ptable_r;
   // apply priors if applicable
   if (app.size() > 0)
      ptable *= real(app(i)(d));
   }

template <class real, class sig, bool norm>
inline real fba2<real, sig, norm>::compute_gamma(int d, int i, int x,
      int deltax) const
   {
   real result = receiver.R(d, i, r.extract(xmax + n * i + x, n + deltax));
   if (app.size() > 0)
      result *= real(app(i)(d));
   return result;
   }

template <class real, class sig, bool norm>
real fba2<real, sig, norm>::get_gamma(int d, int i, int x, int deltax) const
   {
   if (!cache_enabled)
      return compute_gamma(d, i, x, deltax);

   if (!cached[i][x])
      {
      // mark results as cached now
      cached[i][x] = true;
      // allocate space for results
      static array1r_t ptable;
      ptable.init(2 * dxmax + 1);
      // call computation method and store results
      for (int d = 0; d < q; d++)
         {
         compute_gamma(d, i, x, ptable);
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma[d][i][x][deltax] = ptable(dxmax + deltax);
         }
#ifndef NDEBUG
      gamma_misses++;
#endif
      }
#ifndef NDEBUG
   gamma_calls++;
#endif

   return gamma[d][i][x][deltax];
   }

} // end namespace

#endif
