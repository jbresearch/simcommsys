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

#include "lut_interleaver.h"
#include <cstdlib>
#include <iostream>

namespace libcomm {

// transform functions

template <class real>
void lut_interleaver<real>::transform(const libbase::vector<int>& in,
      libbase::vector<int>& out) const
   {
   const int tau = lut.size();
   assertalways(in.size() == tau);
   out.init(in.size());
   for (int t = 0; t < tau; t++)
      if (lut(t) == fsm::tail)
         out(t) = fsm::tail;
      else
         out(t) = in(lut(t));
   }

template <class real>
void lut_interleaver<real>::transform(const libbase::matrix<real>& in,
      libbase::matrix<real>& out) const
   {
   const int tau = lut.size();
   const int K = in.size().cols();
   assertalways(in.size().rows() == tau);
   out.init(in.size());
   for (int t = 0; t < tau; t++)
      if (lut(t) == fsm::tail)
         for (int i = 0; i < K; i++)
            out(t, i) = real(1.0 / K);
      else
         for (int i = 0; i < K; i++)
            out(t, i) = in(lut(t), i);
   }

template <class real>
void lut_interleaver<real>::inverse(const libbase::matrix<real>& in,
      libbase::matrix<real>& out) const
   {
   const int tau = lut.size();
   const int K = in.size().cols();
   assertalways(in.size().rows() == tau);
   out.init(in.size());
   for (int t = 0; t < tau; t++)
      if (lut(t) == fsm::tail)
         for (int i = 0; i < K; i++)
            out(t, i) = real(1.0 / K);
      else
         for (int i = 0; i < K; i++)
            out(lut(t), i) = in(t, i);
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

#define INSTANTIATE(r, x, type) \
   template class lut_interleaver<type>;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
