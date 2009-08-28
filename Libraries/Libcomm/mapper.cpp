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
