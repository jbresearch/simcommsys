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

#ifndef __filter_h
#define __filter_h

#include "config.h"
#include "matrix.h"

namespace libimage {

/*
 * \brief   Filter interface
 * \author  Johann Briffa
 *
 * This class specifies the interface that any filter class should provide.
 * The specification supports tiled filtering through a two-pass process.
 * The first pass gathers details from the image, tile by tile, while the
 * second pass uses the gathered information for any parameter estimates
 * (such as automatic thresholds, etc) and applies the filter to the image.
 */

template <class T>
class filter {
public:
   virtual ~filter()
      {
      }
   // progress display
   virtual void display_progress(const int done, const int total) const = 0;
   // parameter estimation (updates internal statistics)
   virtual void reset() = 0;
   virtual void update(const libbase::matrix<T>& in) = 0;
   virtual void estimate() = 0;
   //! Filter process loop (only updates output matrix)
   virtual void
   process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const = 0;
   //! Apply filter to an image channel
   void apply(const libbase::matrix<T>& in, libbase::matrix<T>& out);
};



} // end namespace

#endif
