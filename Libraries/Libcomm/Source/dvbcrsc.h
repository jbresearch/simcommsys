#ifndef __dvbcrsc_h
#define __dvbcrsc_h

#include "vcs.h"
#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "serializer.h"

/*
  Version 1.00 (13-14 Jul 2006)
  original version, made to conform with fsm 1.50.

  Version 1.10 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.11 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

class dvbcrsc : public fsm {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new dvbcrsc; };
   static const int csct[7][8];  // circulation state correspondence table
   static const int k, n;        // number of input and output bits, respectively
   static const int nu;		      // number of memory elements (constraint length)
   libbase::bitfield reg;  // present state (shift register)
   int N;         // sequence length since last reset;
protected:
   void init();
   dvbcrsc();
public:
   // class management (construction/destruction)
   dvbcrsc(const dvbcrsc& x);		// copy constructor
   ~dvbcrsc();
   
   // class management (cloning/naming)
   dvbcrsc *clone() const { return new dvbcrsc(*this); };		// cloning operation
   const char* name() const { return shelper.name(); };

   // FSM operations (reset/advance/step/state)
   void reset(int state=0);	// resets the FSM to a specified state
   void resetcircular(int zerostate, int n); // resets the FSM, given the zero-state solution and the number of time-steps
   void resetcircular();
   void advance(int& input);  // feeds the specified input and advances the state
   int output(int& input);	   // computes the output for the given input and the present state
   int step(int& input);	// feeds the specified input and returns the corresponding output
   int state() const;		// returns the current state

   // informative functions
   int num_states() const { return 1<<nu; };	// returns the number of defined states
   int num_inputs() const { return 1<<k; };	// returns the number of valid inputs
   int num_outputs() const { return 1<<n; };	// returns the number of valid outputs
   int mem_order() const { return nu; };	   // memory order (length of tail)

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

