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

#ifndef __variancefilter_h
#define __variancefilter_h

#include "filter.h"

/*
 Version 1.00 (30 Nov 2001)
 Initial version - works local variance of matrix, given radius.

 Version 1.10 (17 Oct 2002)
 class is now derived from filter.

 Version 1.20 (10 Nov 2006)
 * defined class and associated data within "libimage" namespace.
 */

namespace libimage {

template <class T>
class variancefilter : public filter<T> {
protected:
   int m_d;
public:
   variancefilter()
      {
      }
   variancefilter(const int d)
      {
      init(d);
      }
   virtual ~variancefilter()
      {
      }
   // initialization
   void init(const int d);
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
