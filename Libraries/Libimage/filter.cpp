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

#include "filter.h"

namespace libimage {

template <class T>
void filter<T>::apply(const libbase::matrix<T>& in,
      libbase::matrix<T>& out)
   {
   // parameter estimation (updates internal statistics)
   reset();
   update(in);
   estimate();
   // filter process loop (only updates output matrix)
   process(in, out);
   }

// Explicit Realizations

template class filter<double> ;
template class filter<float> ;
template class filter<int> ;

} // end namespace
