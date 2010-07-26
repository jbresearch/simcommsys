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
