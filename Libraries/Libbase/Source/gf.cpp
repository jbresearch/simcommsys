#include "gf.h"

#include <stdlib.h>
#include <string>

namespace libbase {

using std::cerr;

// Constructors / Destructors

/*!
   \brief Principal constructor
   \param   n     Order of the binary field extension; that is, the field will be \f$ GF(2^n) \f$.
   \param   poly  Primitive polynomial used to define the field elements
   \param   value Representation of element by its polynomial coefficients
*/
template <int n, int poly> gf<n,poly>::gf(int32u value)
   {
   }

}; // end namespace
