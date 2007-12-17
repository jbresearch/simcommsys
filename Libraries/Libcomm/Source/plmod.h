#ifndef __plmod_h
#define __plmod_h

#include "config.h"

namespace libcomm {

/*!
   \brief   Piece-wise Linear Modulator used in SSIS.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.01 (17 Dec 2007)
   - Moved to libcomm namespace.
*/

double plmod(const double u);

}; // end namespace

#endif
