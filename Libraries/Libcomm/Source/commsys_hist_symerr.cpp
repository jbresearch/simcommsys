/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_hist_symerr.h"
#include <sstream>

namespace libcomm {

// commsys functions

void commsys_hist_symerr::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   const int skip = count()/get_iter();
   int symerrors = countsymerrors(source, decoded);
   // Update the count for that number of symbol errors (may be zero)
   result(skip*i + symerrors)++;
   }

/*!
   \copydoc experiment::result_description()

   The description is a string ER_X_Y, where 'X' is the symbol-error 
   count (starting at zero), and 'Y' is the iteration, starting at 1.
*/
std::string commsys_hist_symerr::result_description(int i) const
   {
   assert(i >= 0 && i < count());
   std::ostringstream sout;
   const int x = i % (get_symbolsperblock()+1);
   const int y = (i / (get_symbolsperblock()+1))+1;
   sout << "ER_" << x << "_" << y;
   return sout.str();
   }

}; // end namespace
