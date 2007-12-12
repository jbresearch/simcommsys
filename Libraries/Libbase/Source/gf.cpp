#include "gf.h"

#include <stdlib.h>
#include <string>

namespace libbase {

using std::cerr;

// Constructors / Destructors

/*!
   \brief Principal constructor
   \param   m     Order of the binary field extension; that is, the field will be \f$ GF(2^m) \f$.
   \param   poly  Primitive polynomial used to define the field elements
   \param   value Representation of element by its polynomial coefficients

   \todo Validate \c poly - this should be a primitive polynomial [cf. Lin & Costello, 2004, p.41]
*/
template <int m, int poly> gf<m,poly>::gf(int32u value)
   {
   assert(value < (1<<m));
   gf::value = value;
   }

}; // end namespace
