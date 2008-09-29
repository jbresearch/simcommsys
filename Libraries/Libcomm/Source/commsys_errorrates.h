#ifndef __commsys_errorrates_h
#define __commsys_errorrates_h

#include "config.h"
#include "vector.h"
#include <string>

namespace libcomm {

/*!
   \brief   CommSys Results - Symbol/Frame Error Rates.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements standard error rate calculators.
*/
class commsys_errorrates {
protected:
   /*! \name System Interface */
   //! The number of decoding iterations performed
   virtual int get_iter() const = 0;
   //! The number of information symbols per block
   virtual int get_symbolsperblock() const = 0;
   //! The information symbol alphabet size
   virtual int get_alphabetsize() const = 0;
   // @}
   /*! \name Helper functions */
   int countbiterrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   int countsymerrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   // @}
public:
   virtual ~commsys_errorrates() {};
   /*! \name Public interface */
   void updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
       For each iteration, we count the number of symbol and frame errors
   */
   int count() const { return 2*get_iter(); };
   int get_multiplicity(int i) const;
   std::string result_description(int i) const;
   // @}
};

}; // end namespace

#endif
