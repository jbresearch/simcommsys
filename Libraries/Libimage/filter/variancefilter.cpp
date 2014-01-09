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

#include "variancefilter.h"
#include "rvstatistics.h"

namespace libimage {

// initialization

template <class T>
void variancefilter<T>::init(const int d)
   {
   m_d = d;
   }

// filter process loop (only updates output matrix)

template <class T>
void variancefilter<T>::process(const libbase::matrix<T>& in,
      libbase::matrix<T>& out) const
   {
   const int M = in.size().rows();
   const int N = in.size().cols();

   out.init(M, N);

   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         {
         // compute the variance of neighbouring pixels
         libbase::rvstatistics r;
         for (int ii = std::max(i - m_d, 0); ii <= std::min(i + m_d, M - 1); ii++)
            for (int jj = std::max(j - m_d, 0); jj <= std::min(j + m_d, N - 1); jj++)
               r.insert(double(in(ii, jj)));
         out(i, j) = T(r.var());
         }
   }

// Explicit Realizations

template class variancefilter<double> ;
template class variancefilter<float> ;
template class variancefilter<int> ;

} // end namespace
