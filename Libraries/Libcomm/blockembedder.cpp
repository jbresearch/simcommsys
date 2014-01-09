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

#include "blockembedder.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// *** Blockwise Data Embedder/Extractor Common Interface ***

// Block modem operations

template <class S, template <class > class C, class dbl>
void basic_blockembedder<S, C, dbl>::embed(const int N, const C<int>& data,
      const C<S>& host, C<S>& stego)
   {
   test_invariant();
   advance_always();
   doembed(N, data, host, stego);
   }

template <class S, template <class > class C, class dbl>
void basic_blockembedder<S, C, dbl>::extract(const channel<S, C>& chan,
      const C<S>& rx, C<array1d_t>& ptable)
   {
   test_invariant();
   advance_if_dirty();
   doextract(chan, rx, ptable);
   mark_as_dirty();
   }

// Explicit Realizations

using libbase::vector;
using libbase::matrix;

template class basic_blockembedder<int, vector, double> ;
template class basic_blockembedder<float, vector, double> ;
template class basic_blockembedder<double, vector, double> ;

template class basic_blockembedder<int, matrix, double> ;
template class basic_blockembedder<float, matrix, double> ;
template class basic_blockembedder<double, matrix, double> ;

// *** Blockwise Data Embedder/Extractor Common Interface ***

// Explicit Realizations

template class blockembedder<int, vector, double> ;
template class blockembedder<float, vector, double> ;
template class blockembedder<double, vector, double> ;

template class blockembedder<int, matrix, double> ;
template class blockembedder<float, matrix, double> ;
template class blockembedder<double, matrix, double> ;

} // end namespace
