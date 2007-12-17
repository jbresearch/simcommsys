/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "named_lut.h"
#include <sstream>

namespace libcomm {

const libbase::vcs named_lut::version("Named LUT Interleaver module (named_lut)", 1.21);

const libbase::serializer named_lut::shelper("interleaver", "named", named_lut::create);

// description output

std::string named_lut::description() const
   {
   std::ostringstream sout;
   sout << "Named Interleaver (" << lutname;
   if(m > 0)
      sout << ", Forced tail length " << m << ")";
   else
      sout << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& named_lut::serialize(std::ostream& sout) const
   {
   sout << m << "\n";
   sout << lutname << "\n";
   sout << lut;
   return sout;
   }

// object serialization - loading

std::istream& named_lut::serialize(std::istream& sin)
   {
   sin >> m;
   sin >> lutname;
   sin >> lut;
   return sin;
   }

}; // end namespace
