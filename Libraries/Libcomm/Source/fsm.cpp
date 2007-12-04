#include "fsm.h"
#include "serializer.h"

namespace libcomm {

const libbase::vcs fsm::version("Finite State Machine module (fsm)", 1.70);

const int fsm::tail = -1;

// FSM operations

int fsm::step(int& input)
   {
   int op = output(input);
   advance(input);
   return op;
   }

// serialization functions

std::ostream& operator<<(std::ostream& sout, const fsm* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, fsm*& x)
   {
   std::string name;
   sin >> name;
   x = (fsm*) libbase::serializer::call("fsm", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (fsm): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
