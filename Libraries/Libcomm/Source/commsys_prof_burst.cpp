/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_prof_burst.h"
#include "fsm.h"
#include <sstream>

namespace libcomm {

// commsys functions

void commsys_prof_burst::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   const int skip = count()/get_iter();
   // Update the relevant count for every symbol in error
   // Check the first symbol first
   assert(source(0) != fsm::tail);
   if(source(0) != decoded(0))
      result(skip*i + 0)++;
   // For each remaining symbol
   for(int t=1; t<get_symbolsperblock(); t++)
      {
      if(source(t-1) != decoded(t-1))
         result(skip*i + 3)++;
      assert(source(t) != fsm::tail);
      if(source(t) != decoded(t))
         {
         // Keep separate counts, depending on whether the previous symbol was in error
         if(source(t-1) != decoded(t-1))
            result(skip*i + 2)++;
         else
            result(skip*i + 1)++;
         }
      }
   }

/*!
   \copydoc experiment::get_multiplicity()

   For each iteration, we count respectively the number symbol errors:
   - in the first frame symbol (at most 1/frame)
   - in subsequent symbols, if the prior symbol was correct
   - in subsequent symbols, if the prior symbol was in error
   - in the prior symbol (required when applying Bayes' rule
     to the above two counts)
     (all three above: at most #symbols/frame - 1)
*/
int commsys_prof_burst::get_multiplicity(int i) const
   {
   assert(i >= 0 && i < count());
   switch(i % 4)
      {
      case 0:
         return 1;
      default:
         return get_symbolsperblock()-1;
      }
   }

/*!
   \copydoc experiment::result_description()

   The description is a string XXX_Y, where 'XXX' is a string indicating
   the probability represented. 'Y' is the iteration, starting at 1.
*/
std::string commsys_prof_burst::result_description(int i) const
   {
   assert(i >= 0 && i < count());
   std::ostringstream sout;
   switch(i % 4)
      {
      case 0:
         sout << "P[e0]_";
         break;
      case 1:
         sout << "P[ei|correct]_";
         break;
      case 2:
         sout << "P[ei|error]_";
         break;
      default:
         sout << "P[ei-1]_";
         break;
      }
   sout << (i/4)+1;
   return sout.str();
   }

}; // end namespace
