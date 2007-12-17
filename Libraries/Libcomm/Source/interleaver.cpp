/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "interleaver.h"
#include "serializer.h"

namespace libcomm {

// serialization functions

std::ostream& operator<<(std::ostream& sout, const interleaver* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, interleaver*& x)
   {
   std::string name;
   sin >> name;
   x = (interleaver*) libbase::serializer::call("interleaver", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (interleaver): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
