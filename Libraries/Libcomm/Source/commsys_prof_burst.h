#ifndef __commsys_prof_burst_h
#define __commsys_prof_burst_h

#include "config.h"
#include "commsys.h"

namespace libcomm {

/*!
   \brief   CommSys Results - Error Burstiness Profile.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (15 Apr 2008)
   - Initial version
   - Determines separately the error probabilities for:
      * the first symbol in a frame
      * a symbol following a correctly-decoded one
      * a symbol following an incorrectly-decoded one
*/

class commsys_prof_burst : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   int count() const { return 3*get_iter(); };
};

}; // end namespace

#endif
