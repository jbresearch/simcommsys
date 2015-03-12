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

#include "limitfilter.h"

namespace libimage {

// initialization

template <class T>
void limitfilter<T>::init(const T lo, const T hi)
   {
   m_lo = lo;
   m_hi = hi;
   }

// filter process loop (only updates output matrix)

template <class T>
void limitfilter<T>::process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const
   {
   const int M = in.size().rows();
   const int N = in.size().cols();

   out.init(M, N);

   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         if (in(i, j) < m_lo)
            out(i, j) = m_lo;
         else if (in(i, j) > m_hi)
            out(i, j) = m_hi;
         else
            out(i, j) = in(i, j);
   }

template <class T>
void limitfilter<T>::process(libbase::matrix<T>& m) const
   {
   const int M = m.size().rows();
   const int N = m.size().cols();

   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         if (m(i, j) < m_lo)
            m(i, j) = m_lo;
         else if (m(i, j) > m_hi)
            m(i, j) = m_hi;
   }

// Explicit Realizations

template class limitfilter<double> ;
template class limitfilter<float> ;
template class limitfilter<int> ;

} // end namespace
