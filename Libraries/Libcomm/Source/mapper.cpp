/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mapper.h"

namespace libcomm {

// Helper functions

template <template<class> class C>
int mapper<C>::get_rate(const int input, const int output)
   {
   const int s = int(round( log(double(output)) / log(double(input)) ));
   assertalways(output == pow(input,s));
   return s;
   }

// Setup functions

template <template<class> class C>
void mapper<C>::set_parameters(const int N, const int M, const int S)
   {
   this->N = N;
   this->M = M;
   this->S = S;
   setup();
   }

// Vector mapper operations

template <template<class> class C>
void mapper<C>::transform(const C<int>& in, C<int>& out) const
   {
   advance_always();
   dotransform(in, out);
   }

template <template<class> class C>
void mapper<C>::inverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   advance_if_dirty();
   doinverse(pin, pout);
   mark_as_dirty();
   }

// Explicit instantiations

template class mapper<libbase::vector>;

}; // end namespace
