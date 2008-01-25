/*!
   \file

   \par Version Control:
   - $Revision: 526 $
   - $Date: 2008-01-24 14:19:25 +0000 (Thu, 24 Jan 2008) $
   - $Author: jabriffa $
*/

#include "bsc.h"
#include <sstream>

namespace libcomm {

const libbase::serializer bsc::shelper("channel", "bsc", bsc::create);

// Channel parameter setters

void bsc::set_parameter(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsc::Ps = Ps;
   }

// Channel function overrides

/*!
   \copydoc channel::corrupt()

   The channel model implemented is described by the following state diagram:
   \dot
   digraph bsidstates {
      // Make figure left-to-right
      rankdir = LR;
      // state definitions
      this [ shape=circle, color=gray, style=filled, label="t(i)" ];
      next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
      // path definitions
      this -> next [ label="1-Ps" ];
      this -> Substitute [ label="Ps" ];
      Substitute -> next;
   }
   \enddot
*/
bool bsc::corrupt(const bool& s)
   {
   const double p = r.fval();
   if(p < Ps)
      return !s;
   return s;
   }

// description output

std::string bsc::description() const
   {
   return "BSC channel";
   }

}; // end namespace