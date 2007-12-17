/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "flat.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

const libbase::serializer flat::shelper("interleaver", "flat", flat::create);

// initialization

void flat::init(const int tau)
   {
   lut.init(tau);
   for(int i=0; i<tau; i++)
      lut(i) = i;
   }

// description output

std::string flat::description() const
   {
   std::ostringstream sout;
   sout << "Flat Interleaver";
   return sout.str();
   }

// object serialization - saving

std::ostream& flat::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   return sout;
   }

// object serialization - loading

std::istream& flat::serialize(std::istream& sin)
   {
   int tau;
   sin >> tau;
   init(tau);
   return sin;
   }

}; // end namespace
