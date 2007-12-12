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
*/
template <int n, int poly> gf<n,poly>::gf()
   {
   }

}; // end namespace
