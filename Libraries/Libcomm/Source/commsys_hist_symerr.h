#ifndef __commsys_hist_symerr_h
#define __commsys_hist_symerr_h

#include "config.h"
#include "commsys_errorrates.h"

namespace libcomm {

/*!
   \brief   CommSys Results - Symbol-Error per Frame Histogram.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Computes histogram of symbol error count for each block simulated.
*/

class commsys_hist_symerr : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
       For each iteration, we count the frequency of each possible
       symbol-error count, including zero
   */
   int count() const { return (get_symbolsperblock()+1)*get_iter(); };
   /*! \copydoc experiment::get_multiplicity()
       Only one result can be incremented for every frame.
   */
   int get_multiplicity(int i) const { return 1; };
   std::string result_description(int i) const;
};

}; // end namespace

#endif
