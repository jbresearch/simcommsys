#ifndef __dminner2d_h
#define __dminner2d_h

#include "config.h"

#include "dminner2.h"
#include "fba2.h"

namespace libcomm {

/*!
   \brief   Iterative 2D implementation of Davey-MacKay Inner Code.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements a 2D version of the Davey-MacKay inner code, using iterative
   row/column decoding, where the sparse symbols are now two-dimensional.
*/

template <class real, bool normalize>
class dminner2d {
public:
   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dminner2d);
};

}; // end namespace

#endif
