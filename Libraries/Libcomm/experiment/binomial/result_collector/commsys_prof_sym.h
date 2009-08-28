#ifndef __commsys_prof_sym_h
#define __commsys_prof_sym_h

#include "config.h"
#include "commsys_errorrates.h"

namespace libcomm {

/*!
 * \brief   CommSys Results - Symbol-Value Error Profile.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Computes symbol-error histogram as dependent on source symbol value.
 */

class commsys_prof_sym : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i,
         const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count the number of symbol errors for
    * every input alphabet symbol value.
    */
   int count() const
      {
      return get_alphabetsize() * get_iter();
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
