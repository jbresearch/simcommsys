#ifndef __gf_h
#define __gf_h

#include "config.h"

#include <iostream>
#include <string>

namespace libbase {

/*!
   \brief   Galois Field.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (11 Dec 2007)
   - Initial version; implements extensions of the binary field: \f$ GF(2^n) \f$.
   - This is the first class where we're not using the vcs version-printing class.
*/

template <int n, int poly> class gf {
private:
public:
   /*! \name Constructors / Destructors */
   gf();
   // @}

};

}; // end namespace

#endif
