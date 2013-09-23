/*!
 * \file
 * $Id$
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

#ifndef __fba2_interface_h
#define __fba2_interface_h

#include "config.h"
#include "vector.h"
#include "instrumented.h"

#include <string>

namespace libcomm {

/*!
 * \brief   Interface for Symbol-Level Forward-Backward Algorithm (for TVB codes).
 * \author  Johann Briffa
 * $Id$
 *
 * Defines the interface for the forward-backward algorithm for a HMM, as
 * required for the MAP decoding algorithm for a generalized class of
 * synchronization-correcting codes described in
 * Briffa et al, "A MAP Decoder for a General Class of Synchronization-
 * Correcting Codes", Submitted to Trans. IT, 2011.
 *
 * \tparam receiver_t Type for receiver metric computer
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 *
 * \todo Extract interface for receiver type
 */

template <class receiver_t, class sig, class real>
class fba2_interface {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   // @}
public:
   //! Determine memory required for global storage mode (in MiB)
   static int get_memory_required(int N, int n, int q, int mtau_min,
         int mtau_max, int mn_min, int mn_max)
      {
      // determine memory required
      // NOTE: do all computations at 64-bit, or we get intermediate overflow!
      libbase::int64u bytes_required = sizeof(real);
      bytes_required *= q;
      bytes_required *= N;
      bytes_required *= (mtau_max - mtau_min + 1);
      bytes_required *= (mn_max - mn_min + 1);
      bytes_required >>= 20;
      return int(bytes_required);
      }

   /*! \name Interface with derived classes */
   //! Main initialization routine
   virtual void init(int N, int n, int q, int mtau_min, int mtau_max,
         int mn_min, int mn_max, int m1_min, int m1_max, double th_inner,
         double th_outer) = 0;
   //! Access metric computation
   virtual receiver_t& get_receiver() const = 0;

   // decode functions
   virtual void decode(libcomm::instrumented& collector, const array1s_t& r,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post, const int offset) = 0;
   virtual void get_drift_pdf(array1r_t& pdf, const int i) const = 0;
   virtual void get_drift_pdf(array1vr_t& pdftable) const = 0;

   //! Description
   virtual std::string description() const = 0;
   // @}
};

} // end namespace

#endif
