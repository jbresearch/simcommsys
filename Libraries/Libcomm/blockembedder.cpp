/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
