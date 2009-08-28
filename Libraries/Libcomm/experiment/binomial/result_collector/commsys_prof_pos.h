#ifndef __commsys_prof_pos_h
#define __commsys_prof_pos_h

#include "config.h"
#include "commsys_errorrates.h"

namespace libcomm {

/*!
 * \brief   CommSys Results - Frame-Position Error Profile.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Profiler of error with respect to position within block.
 */

class commsys_prof_pos : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i,
         const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count the number of symbol errors for
    * every frame position.
    */
   int count() const
      {
      return get_symbolsperblock() * get_iter();
      }
   /*! \copydoc experiment::get_multiplicity()
    * A total equal to the number of symbols/frame may be incremented
    * in every sample.
    */
   int get_multiplicity(int i) const
      {
      return get_symbolsperblock();
      }
   std::string result_description(int i) const;
};

} // end namespace

#endif
