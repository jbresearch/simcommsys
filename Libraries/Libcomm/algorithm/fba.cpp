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

#include "fba.h"
#include "pacifier.h"
#include <iomanip>

namespace libcomm {

/*! \brief Memory allocator for working matrices
 */
template <class real, class sig, bool norm>
void fba<real, sig, norm>::allocate()
   {
   // Allocate required size
   // F needs indices (j,y) where j in [0, tau-1] and y in [-xmax, xmax]
   // B needs indices (j,y) where j in [1, tau] and y in [-xmax, xmax]
   typedef boost::multi_array_types::extent_range range;
   F.resize(boost::extents[tau][range(-xmax, xmax + 1)]);
   B.resize(boost::extents[range(1, tau + 1)][range(-xmax, xmax + 1)]);
   // flag the state of the arrays
   initialised = true;

   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const int prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(real) * (F.num_elements()
         + B.num_elements());
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB" << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
   }

/*! \brief Release memory for working matrices
 */
template <class real, class sig, bool norm>
void fba<real, sig, norm>::free()
   {
   F.resize(boost::extents[0][0]);
   B.resize(boost::extents[0][0]);
   // flag the state of the arrays
   initialised = false;
   }

// Initialization

template <class real, class sig, bool norm>
void fba<real, sig, norm>::init(int tau, int I, int xmax, double th_inner)
   {
   // if any parameters that effect memory have changed, release memory
   if (initialised && (tau != This::tau || xmax != This::xmax))
      free();
   // code parameters
   assert(tau > 0);
   This::tau = tau;
   // decoder parameters
   assert(I > 0);
   assert(xmax > 0);
   This::I = I;
   This::xmax = xmax;
   // path truncation parameters
   assert(th_inner >= 0 && th_inner <= 1);
   This::th_inner = real(th_inner);
   }

// Internal procedures

template <class real, class sig, bool norm>
void fba<real, sig, norm>::work_forward(const array1s_t& r)
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
         for (index a = -xmax; a <= xmax; ++a)
            if (F[j - 1][a] > threshold)
               threshold = F[j - 1][a];
         threshold *= th_inner;
         }
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      const index amin = std::max(-xmax, 1 - int(j));
      const index amax = xmax;
      const index ymax_bnd = std::min(xmax, r.size() - int(j));
      for (index a = amin; a <= amax; ++a)
         {
         // ignore paths below a certain threshold
         if (thresholding && F[j - 1][a] < threshold)
            continue;
         const index ymin = std::max(-xmax, int(a) - 1);
         const index ymax = std::min(int(ymax_bnd), int(a) + I);
         for (index y = ymin; y <= ymax; ++y)
            F[j][y] += F[j - 1][a] * R(int(j - 1), r.extract(int(j - 1 + a),
                  int(y - a + 1)));
         }
      // normalize if requested
      if (norm)
         {
         real scale = 0;
         for (index y = -xmax; y <= xmax; ++y)
            scale += F[j][y];
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (index y = -xmax; y <= xmax; ++y)
            F[j][y] *= scale;
         }
      }
   std::cerr << progress.update(tau - 1, tau - 1);
   }

template <class real, class sig, bool norm>
void fba<real, sig, norm>::work_backward(const array1s_t& r)
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
   assertalways(abs(r.size()-tau) <= xmax);
   B[tau][r.size() - tau] = real(1);
   // compute remaining matrix values
   for (index j = tau - 1; j > 0; --j)
      {
      std::cerr << progress.update(int(tau - 1 - j), tau - 1);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (index b = -xmax; b <= xmax; ++b)
            if (B[j + 1][b] > threshold)
               threshold = B[j + 1][b];
         threshold *= th_inner;
         }
      // event must fit the received sequence:
      // 1. j+y >= 0
      // 2. j+b <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. b-y <= I
      // 4. b-y >= -1
      const index bmin = -xmax;
      const index bmax = std::min(xmax, r.size() - int(j) - 1);
      const index ymin_bnd = std::max(-xmax, int(-j));
      for (index b = bmin; b <= bmax; ++b)
         {
         // ignore paths below a certain threshold
         if (thresholding && B[j + 1][b] < threshold)
            continue;
         const index ymin = std::max(int(ymin_bnd), int(b) - I);
         const index ymax = std::min(xmax, int(b) + 1);
         for (index y = ymin; y <= ymax; ++y)
            B[j][y] += B[j + 1][b] * R(int(j), r.extract(int(j + y), int(b - y
                  + 1)));
         }
      // normalize if requested
      if (norm)
         {
         real scale = 0;
         for (index y = -xmax; y <= xmax; ++y)
            scale += B[j][y];
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (index y = -xmax; y <= xmax; ++y)
            B[j][y] *= scale;
         }
      }
   std::cerr << progress.update(tau - 1, tau - 1);
   }

// User procedures

template <class real, class sig, bool norm>
void fba<real, sig, norm>::prepare(const array1s_t& r)
   {
   // compute forwards and backwards passes
   work_forward(r);
   work_backward(r);
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

template class fba<logrealfast, bool, false> ;
template class fba<double, bool, true> ;
template class fba<float, bool, true> ;

} // end namespace
