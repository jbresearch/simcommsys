#ifndef __symbol_h
#define __symbol_h

#include "config.h"

namespace libcomm {

/*!
   \brief   Modulation symbol.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (22 Jan 2008)
   - Initial version.
   - Created to abstract the concept of a modulation symbol from its signal-space
     representation.
*/

class symbol {
public:
   /*! \name Constructors / Destructors */
   symbol() {};
   virtual ~symbol() {};
   // @}
};

}; // end namespace

#endif
