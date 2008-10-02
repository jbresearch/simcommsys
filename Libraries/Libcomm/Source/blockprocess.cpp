/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "blockprocess.h"

namespace libcomm {

// Block-processing operations

void blockprocess::advance_always() const
   {
   advance();
   advanced = true;
   }

void blockprocess::advance_if_dirty() const
   {
   if(!advanced)
      {
      advance();
      advanced = true;
      }
   }

}; // end namespace
