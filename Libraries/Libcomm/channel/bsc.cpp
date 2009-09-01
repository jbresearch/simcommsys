/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "bsc.h"
#include <sstream>

namespace libcomm {

const libbase::serializer bsc::shelper("channel", "bsc", bsc::create);

// Channel parameter handling

void bsc::set_parameter(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsc::Ps = Ps;
   }

// Channel function overrides

/*!
 * \copydoc channel::corrupt()
 * 
 * The channel model implemented is described by the following state diagram:
 * \dot
 * digraph bsidstates {
 * // Make figure left-to-right
 * rankdir = LR;
 * // state definitions
 * this [ shape=circle, color=gray, style=filled, label="t(i)" ];
 * next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
 * // path definitions
 * this -> next [ label="1-Ps" ];
 * this -> Substitute [ label="Ps" ];
 * Substitute -> next;
 * }
 * \enddot
 */
bool bsc::corrupt(const bool& s)
   {
   const double p = r.fval_closed();
   if (p < Ps)
      return !s;
   return s;
   }

// description output

std::string bsc::description() const
   {
   return "BSC channel";
   }

// object serialization - saving

std::ostream& bsc::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& bsc::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace
