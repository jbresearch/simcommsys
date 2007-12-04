#include "experiment.h"
#include "serializer.h"

namespace libcomm {

const libbase::vcs experiment::version("Generic experiment module (experiment)", 1.30);

// serialization functions

std::ostream& operator<<(std::ostream& sout, const experiment* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, experiment*& x)
   {
   std::string name;
   sin >> name;
   x = (experiment*) libbase::serializer::call("experiment", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (experiment): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
