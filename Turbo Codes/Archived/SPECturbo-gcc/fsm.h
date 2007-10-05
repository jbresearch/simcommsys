#ifndef __fsm_h
#define __fsm_h

#include "vcs.h"
#include <iostream.h>

extern const vcs fsm_version;

class fsm {
public:
   static const int tail;				// a special input to use when tailing out
   
   virtual fsm *clone() = 0;			// cloning operation
   virtual ~fsm() {};					// virtual destructor
   
   virtual void reset(int state=0) = 0;	// resets the FSM to a specified state
   virtual int step(int& input) = 0;		// feeds the specified input and returns the corresponding output
   virtual int state() const = 0;		   // returns the current state
   virtual int num_states() const = 0;	   // returns the number of defined states
   virtual int num_inputs() const = 0;	   // returns the number of valid inputs
   virtual int num_outputs() const = 0;	// returns the number of valid outputs
   virtual int mem_order() const = 0;	   // memory order (length of tail)

   virtual void print(ostream& s) const = 0;
};

#endif

