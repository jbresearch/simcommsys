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

#ifndef __atmfilter_h
#define __atmfilter_h

#include "filter.h"

namespace libimage {

/*
 * \brief   Alpha-Trimmed Mean Filter
 * \author  Johann Briffa
 *
 * This filter computes the alpha-trimmed mean within a given neighbourhood.
 */

template <class T>
class atmfilter : public filter<T> {
protected:
   int m_d; //!< greatest distance from current pixel in neighbourhood
   int m_alpha; //!< number of outliers to trim at each end before computing mean
public:
   atmfilter(const int d, const int alpha)
      {
      init(d, alpha);
      }
   // initialization
   void init(const int d, const int alpha);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset()
      {
      }
   void update(const libbase::matrix<T>& in)
      {
      }
   void estimate()
      {
      }
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
};

} // end namespace

#endif
