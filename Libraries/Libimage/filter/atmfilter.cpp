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

#include "atmfilter.h"
#include <list>
#include <algorithm>
#include <numeric>

namespace libimage {

// initialization

template <class T>
void atmfilter<T>::init(const int d, const int alpha)
   {
   m_d = d;
   m_alpha = alpha;
   }

// filter process loop (only updates output matrix)

template <class T>
void atmfilter<T>::process(const libbase::matrix<T>& in,
      libbase::matrix<T>& out) const
   {
   const int M = in.size().rows();
   const int N = in.size().cols();

   out.init(M, N);
   using std::list;
   list<T> lst;

   for (int i = 0; i < M; i++)
      {
      display_progress(i, M);
      for (int j = 0; j < N; j++)
         {
         // create list of neighbouring pixels
         lst.clear();
         for (int ii = std::max(i - m_d, 0); ii <= std::min(i + m_d, M - 1); ii++)
            for (int jj = std::max(j - m_d, 0); jj <= std::min(j + m_d, N - 1); jj++)
               lst.push_back(in(ii, jj));
         // sort the list
         lst.sort();
         // erase the first and last alpha elements
         typename list<T>::iterator p1 = lst.begin();
         typename list<T>::iterator p2 = lst.end();
         for (int k = 0; k < m_alpha; k++)
            {
            p1++;
            p2--;
            }
         // compute the mean, skipping the first and last alpha elements
         const int n = lst.size() - 2 * m_alpha;
         T d = 0;
         d = accumulate(p1, p2, d);
         out(i, j) = d / n;
         }
      }
   }

// Explicit Realizations

template class atmfilter<double> ;
template class atmfilter<float> ;
template class atmfilter<int> ;

} // end namespace
