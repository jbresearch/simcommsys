/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "fsm.h"
#include "serializer.h"

namespace libcomm {

const int fsm::tail = -1;

// FSM state operations

void fsm::reset(int state)
   {
   N = 0;
   }

void fsm::resetcircular()
   {
   resetcircular(state(),N);
   }

// FSM operations

void fsm::advance(int& input)
   {
   N++;
   }

int fsm::step(int& input)
   {
   int op = output(input);
   advance(input);
   return op;
   }

// stream output operators

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
