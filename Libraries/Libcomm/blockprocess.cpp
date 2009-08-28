/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "blockprocess.h"

namespace libcomm {

// Block-processing operations

void blockprocess::advance_always() const
   {
   advance();
   dirty = false;
   }

void blockprocess::advance_if_dirty() const
   {
   if (dirty)
      {
      advance();
      dirty = false;
      }
   }

} // end namespace
