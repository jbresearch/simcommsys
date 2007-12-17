/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "codec.h"
#include "serializer.h"

namespace libcomm {

// serialization functions

std::ostream& operator<<(std::ostream& sout, const codec* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, codec*& x)
   {
   std::string name;
   sin >> name;
   x = (codec*) libbase::serializer::call("codec", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (codec): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
