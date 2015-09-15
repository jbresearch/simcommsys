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

#include "fba.h"
#include "pacifier.h"
#include <iomanip>

namespace libcomm {

/*! \brief Memory allocator for working matrices
 */
template <class sig, class real>
void fba<sig, real>::allocate()
   {
   // Allocate required size
   // F needs indices (j,y) where j in [0, tau-1] and y in [mtau_min, mtau_max]
   // B needs indices (j,y) where j in [1, tau] and y in [mtau_min, mtau_max]
   typedef boost::multi_array_types::extent_range range;
   F.resize(boost::extents[tau][range(mtau_min, mtau_max + 1)]);
   B.resize(boost::extents[range(1, tau + 1)][range(mtau_min, mtau_max + 1)]);
   // flag the state of the arrays
   initialised = true;

   // if this is not the first time, skip the rest
   static bool first_time = true;
   if (!first_time)
      return;
   first_time = false;

#ifndef NDEBUG
   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const std::streamsize prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(real) * (F.num_elements()
         + B.num_elements());
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB" << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class sig, class real>
void fba<sig, real>::free()
   {
   F.resize(boost::extents[0][0]);
   B.resize(boost::extents[0][0]);
   // flag the state of the arrays
   initialised = false;
   }

// Initialization

template <class sig, class real>
void fba<sig, real>::init(int tau, int mtau_min, int mtau_max, int m1_min, int m1_max, double th_inner, bool norm)
   {
   // if any parameters that effect memory have changed, release memory
   if (initialised
         && (tau != this->tau || mtau_min != this->mtau_min
               || mtau_max != this->mtau_max))
      free();
   // code parameters
   assert(tau > 0);
   this->tau = tau;
   // decoder parameters
   assert(mtau_min <= 0);
   assert(mtau_max >= 0);
   this->mtau_min = mtau_min;
   this->mtau_max = mtau_max;
   assert(m1_min <= 0);
   assert(m1_max >= 0);
   this->m1_min = m1_min;
   this->m1_max = m1_max;
   // path truncation parameters
   assert(th_inner >= 0 && th_inner <= 1);
   this->th_inner = real(th_inner);
   // decoding mode parameters
   this->norm = norm;
   }

// Internal procedures

template <class sig, class real>
void fba<sig, real>::work_forward(const array1s_t& r)
   {
   libbase::pacifier progress("FBA Forward Pass");
   // local flag for path thresholding
   const bool thresholding = (th_inner > real(0));
   // initialise memory if necessary
   if (!initialised)
      allocate();
   // initialise array:
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   typedef typename array2r_t::index index;
   F = real(0);
   F[0][0] = real(1);
   // compute remaining matrix values
   for (index j = 1; j < tau; ++j)
      {
      std::cerr << progress.update(int(j - 1), tau - 1);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (index a = mtau_min; a <= mtau_max; ++a)
            if (F[j - 1][a] > threshold)
               threshold = F[j - 1][a];
         threshold *= th_inner;
         }
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. y-a <= m1_max
      // 4. y-a >= m1_min
      const index amin = std::max(mtau_min, 1 - int(j));
      const index amax = mtau_max;
      const index ymax_bnd = std::min(mtau_max, r.size() - int(j));
      for (index a = amin; a <= amax; ++a)
         {
         // ignore paths below a certain threshold
         if (thresholding && F[j - 1][a] < threshold)
            continue;
         const index ymin = std::max(mtau_min, int(a) + m1_min);
         const index ymax = std::min(int(ymax_bnd), int(a) + m1_max);
         for (index y = ymin; y <= ymax; ++y)
            F[j][y] += F[j - 1][a] * R(int(j - 1), r.extract(int(j - 1 + a),
                  int(y - a + 1)));
         }
      // normalize if requested
      if (norm)
         {
         real scale = 0;
         for (index y = mtau_min; y <= mtau_max; ++y)
            scale += F[j][y];
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (index y = mtau_min; y <= mtau_max; ++y)
            F[j][y] *= scale;
         }
      }
   std::cerr << progress.update(tau - 1, tau - 1);
   }

template <class sig, class real>
void fba<sig, real>::work_backward(const array1s_t& r)
   {
   libbase::pacifier progress("FBA Backward Pass");
   // local flag for path thresholding
   const bool thresholding = (th_inner > real(0));
   // initialise memory if necessary
   if (!initialised)
      allocate();
   // initialise array:
   // we also know x[tau] = r.size()-tau;
   // ie. drift before transmitting bit t[tau] is the discrepancy in the received vector size from tau
   typedef typename array2r_t::index index;
   B = real(0);
   assertalways(abs(r.size()-tau) <= mtau_max);
   B[tau][r.size() - tau] = real(1);
   // compute remaining matrix values
   for (index j = tau - 1; j > 0; --j)
      {
      std::cerr << progress.update(int(tau - 1 - j), tau - 1);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (index b = mtau_min; b <= mtau_max; ++b)
            if (B[j + 1][b] > threshold)
               threshold = B[j + 1][b];
         threshold *= th_inner;
         }
      // event must fit the received sequence:
      // 1. j+y >= 0
      // 2. j+b <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. b-y <= m1_max
      // 4. b-y >= m1_min
      const index bmin = mtau_min;
      const index bmax = std::min(mtau_max, r.size() - int(j) - 1);
      const index ymin_bnd = std::max(mtau_min, int(-j));
      for (index b = bmin; b <= bmax; ++b)
         {
         // ignore paths below a certain threshold
         if (thresholding && B[j + 1][b] < threshold)
            continue;
         const index ymin = std::max(int(ymin_bnd), int(b) - m1_max);
         const index ymax = std::min(mtau_max, int(b) - m1_min);
         for (index y = ymin; y <= ymax; ++y)
            B[j][y] += B[j + 1][b] * R(int(j), r.extract(int(j + y), int(b - y
                  + 1)));
         }
      // normalize if requested
      if (norm)
         {
         real scale = 0;
         for (index y = mtau_min; y <= mtau_max; ++y)
            scale += B[j][y];
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (index y = mtau_min; y <= mtau_max; ++y)
            B[j][y] *= scale;
         }
      }
   std::cerr << progress.update(tau - 1, tau - 1);
   }

// User procedures

template <class sig, class real>
void fba<sig, real>::prepare(const array1s_t& r)
   {
   // compute forwards and backwards passes
   work_forward(r);
   work_backward(r);
   }

} // end namespace

#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/enum.hpp>

using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

#define INSTANTIATE(r, x, type) \
      template class fba<bool, type> ;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
