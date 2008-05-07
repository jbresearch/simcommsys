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
   /*! \copydoc experiment::count()
       For each iteration, we count respectively the number symbol errors:
       - in the first frame symbol
       - in subsequent symbols, if the prior symbol was correct
       - in subsequent symbols, if the prior symbol was in error
       - in the prior symbol (required when applying Bayes' rule
         to the above two counts)
   */
   int count() const { return 4*get_iter(); };
   int get_multiplicity(int i) const;
};

}; // end namespace

#endif
