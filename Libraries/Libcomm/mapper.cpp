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
 * 
 * \section svn Version Control
 * - $Id$
 */

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "mapper.h"

namespace libcomm {

// Helper functions

template <template <class > class C, class dbl>
int mapper<C, dbl>::get_rate(const int input, const int output)
   {
   const int s = int(round(log(double(output)) / log(double(input))));
   assertalways(output == pow(input,s));
   return s;
   }

// Setup functions

template <template <class > class C, class dbl>
void mapper<C, dbl>::set_parameters(const int N, const int M, const int S)
   {
   this->N = N;
   this->M = M;
   this->S = S;
   setup();
   }

// Vector mapper operations

template <template <class > class C, class dbl>
void mapper<C, dbl>::transform(const C<int>& in, C<int>& out) const
   {
   advance_always();
   dotransform(in, out);
   }

template <template <class > class C, class dbl>
void mapper<C, dbl>::inverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   advance_if_dirty();
   doinverse(pin, pout);
   mark_as_dirty();
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;

template class mapper<vector> ;
template class mapper<vector, float> ;
template class mapper<vector, logrealfast> ;
template class mapper<matrix> ;
template class mapper<matrix, float> ;
template class mapper<matrix, logrealfast> ;
} // end namespace
