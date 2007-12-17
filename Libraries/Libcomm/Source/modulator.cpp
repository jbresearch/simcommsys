/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "modulator.h"
#include "serializer.h"
#include "itfunc.h"
#include <stdlib.h>

namespace libcomm {

// serialization functions

std::ostream& operator<<(std::ostream& sout, const modulator* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, modulator*& x)
   {
   std::string name;
   sin >> name;
   x = (modulator*) libbase::serializer::call("modulator", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (modulator): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
