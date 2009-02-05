/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "symbol.h"

namespace libcomm {

// Internal functions

/*!
   \brief Initialization
   \param   value Integer representation of element
*/
template <int q>
void finite_symbol<q>::init(int value)
   {
   assert(value >=0 && value < (1<<q));
   finite_symbol::value = value;
   }

}; // end namespace
