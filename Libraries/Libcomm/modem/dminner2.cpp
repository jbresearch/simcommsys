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

#include "dminner2.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// implementations of channel-specific metrics for fba2

#ifndef USE_CUDA
template <class real, bool norm>
real dminner2<real, norm>::R(int d, int i, const array1b_t& r) const
   {
   const int n = Base::n;
   const int w = Base::ws(i);
   const int s = Base::lut(d);
   // 'tx' is the vector of transmitted symbols that we're considering
   array1b_t tx(n);
   // NOTE: we transmit the low-order bits first
   for (int bit = 0, t = w ^ s; bit < n; bit++, t >>= 1)
      tx(bit) = (t & 1);
   // compute the conditional probability
   return Base::mychan.receive(tx, r);
   }
#endif

// Setup procedure

template <class real, bool norm>
void dminner2<real, norm>::init(const channel<bool>& chan, const int rho)
   {
   // Inherit block size from last modulation step
   const int q = 1 << Base::k;
   const int n = Base::n;
   const int N = Base::ws.size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   Base::mychan = dynamic_cast<const bsid&> (chan);
   // Set channel block size to q-ary symbol size
   Base::mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const int I = Base::mychan.compute_I(tau);
   const int xmax = std::max(Base::mychan.compute_xmax(tau), abs(rho - tau));
   const int dxmax = Base::mychan.compute_xmax(n);
   Base::checkforchanges(I, xmax);
#ifdef USE_CUDA
   // Initialize forward-backward algorithm
   fba.init(N, n, q, I, xmax, dxmax, Base::th_inner, Base::th_outer);
   // initialize our embedded metric computer with unchanging elements
   fba.get_receiver().init(n, Base::lut, Base::mychan);
#else
   FBA::init(N, n, q, I, xmax, dxmax, Base::th_inner, Base::th_outer);
#endif
   }

template <class real, bool norm>
void dminner2<real, norm>::advance() const
   {
   // advance the base class
   Base::advance();
#ifdef USE_CUDA
   // initialize our embedded metric computer
   fba.get_receiver().init(Base::ws);
#endif
   }

// encoding and decoding functions

template <class real, bool norm>
void dminner2<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, array1vd_t& ptable)
   {
   init(chan, rx.size().length());
   array1vr_t p;
#ifdef USE_CUDA
   fba.decode(rx, p);
#else
   FBA::decode(rx, p);
#endif
   Base::normalize_results(p, ptable);
   }

template <class real, bool norm>
void dminner2<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   init(chan, rx.size().length());
   array1vr_t p;
#ifdef USE_CUDA
   fba.decode(rx, app, p);
#else
   FBA::decode(rx, app, p);
#endif
   Base::normalize_results(p, ptable);
   }

// description output

template <class real, bool norm>
std::string dminner2<real, norm>::description() const
   {
   std::ostringstream sout;
   sout << "Symbol-level " << Base::description();
#ifdef USE_CUDA
   sout << ", " << fba.description();
#else
   sout << ", " << FBA::description();
#endif
   return sout.str();
   }

// object serialization - saving

template <class real, bool norm>
std::ostream& dminner2<real, norm>::serialize(std::ostream& sout) const
   {
   return Base::serialize(sout);
   }

// object serialization - loading

template <class real, bool norm>
std::istream& dminner2<real, norm>::serialize(std::istream& sin)
   {
   return Base::serialize(sin);
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;
using libbase::serializer;

#ifndef USE_CUDA
template class dminner2<logrealfast, false> ;
template <>
const serializer dminner2<logrealfast, false>::shelper = serializer(
      "blockmodem", "dminner2<logrealfast>",
      dminner2<logrealfast, false>::create);
#endif

template class dminner2<double, true> ;
template <>
const serializer dminner2<double, true>::shelper = serializer("blockmodem",
      "dminner2<double>", dminner2<double, true>::create);

template class dminner2<float, true> ;
template <>
const serializer dminner2<float, true>::shelper = serializer("blockmodem",
      "dminner2<float>", dminner2<float, true>::create);

} // end namespace
