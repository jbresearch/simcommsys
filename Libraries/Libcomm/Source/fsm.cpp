/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "fsm.h"

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

}; // end namespace
