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

#include "awfilter.h"

namespace libimage {

// initialization

template <class T>
void awfilter<T>::init(const int d, const double noise)
   {
   m_d = d;
   m_autoestimate = false;
   m_noise = noise;
   }

template <class T>
void awfilter<T>::init(const int d)
   {
   m_d = d;
   m_autoestimate = true;
   m_noise = 0;
   }

// parameter estimation (updates internal statistics)

template <class T>
void awfilter<T>::reset()
   {
   if (m_autoestimate)
      rvglobal.reset();
   }

template <class T>
void awfilter<T>::update(const libbase::matrix<T>& in)
   {
   if (!m_autoestimate)
      return;

   const int M = in.size().rows();
   const int N = in.size().cols();

   for (int i = 0; i < M; i++)
      {
      display_progress(i, M);
      for (int j = 0; j < N; j++)
         {
         // compute mean and variance of neighbouring pixels
         libbase::rvstatistics rv;
         for (int ii = std::max(i - m_d, 0); ii <= std::min(i + m_d, M - 1); ii++)
            for (int jj = std::max(j - m_d, 0); jj <= std::min(j + m_d, N - 1); jj++)
               rv.insert(in(ii, jj));
         // add to the global sum
         rvglobal.insert(rv.var());
         }
      }
   }

template <class T>
void awfilter<T>::estimate()
   {
   if (m_autoestimate)
      {
      m_noise = rvglobal.mean();
      libbase::trace << "Noise threshold = " << m_noise << std::endl;
      }
   }

// filter process loop (only updates output matrix)

template <class T>
void awfilter<T>::process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const
   {
   const int M = in.size().rows();
   const int N = in.size().cols();

   out.init(M, N);

   for (int i = 0; i < M; i++)
      {
      display_progress(i, M);
      for (int j = 0; j < N; j++)
         {
         // compute mean and variance of neighbouring pixels
         libbase::rvstatistics rv;
         for (int ii = std::max(i - m_d, 0); ii <= std::min(i + m_d, M - 1); ii++)
            for (int jj = std::max(j - m_d, 0); jj <= std::min(j + m_d, N - 1); jj++)
               rv.insert(in(ii, jj));
         const double mean = rv.mean();
         const double var = rv.var();
         // compute result
         out(i, j) = T(mean + (std::max<double>(0, var - m_noise) / std::max<
               double>(var, m_noise)) * (in(i, j) - mean));
         }
      }
   }

// Explicit Realizations
template class awfilter<double> ;
template class awfilter<float> ;
template class awfilter<int> ;

} // end namespace
