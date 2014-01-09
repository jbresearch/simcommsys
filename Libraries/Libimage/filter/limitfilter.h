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

#ifndef __limitfilter_h
#define __limitfilter_h

#include "filter.h"

namespace libimage {

/*
 * \brief   Limit Filter
 * \author  Johann Briffa
 *
 * This filter limits pixels between given values.
 */

template <class T>
class limitfilter : public filter<T> {
protected:
   T m_lo;
   T m_hi;
public:
   limitfilter()
      {
      }
   limitfilter(const T lo, const T hi)
      {
      init(lo, hi);
      }
   virtual ~limitfilter()
      {
      }
   // initialization
   void init(const T lo, const T hi);
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
   void process(libbase::matrix<T>& m) const;
};

} // end namespace

#endif
