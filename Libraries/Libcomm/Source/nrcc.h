#ifndef __nrcc_h
#define __nrcc_h

#include "vcs.h"
#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "serializer.h"

/*
  Version 1.01 (4 Nov 2001)
  added a function which outputs details on the finite state machine (in accordance 
  with fsm 1.10)

  Version 1.10 (28 Feb 2002)
  added serialization facility (in accordance with fsm 1.20)

  Version 1.11 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.
  also modified system to use '+' instead of ',' for concatenating bitfields, to
  conform with bitfield 1.11.

  Version 1.12 (11 Mar 2002)
  changed the definition of the cloning operation to be a const member, to conform with
  fsm 1.22. Also, made that function inline.

  Version 1.20 (11 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with fsm 1.30), by adding the necessary name() function, and also by removing the class
  name reading/writing in serialize(); this is now done only in the stream << and >>
  functions. serialize() assumes that the correct class is being read/written. We also
  add a static serializer member and initialize it with this class's name and the static
  constructor (adding that too, together with the necessary private empty constructor).
  Also made the fsm object a public base class, rather than a virtual public one, since
  this was affecting the transfer of virtual functions within the class (causing access
  violations). Also changed the access to the init() function to protected, which should
  make deriving from this class a bit easier.

  Version 1.21 (14 Mar 2002)
  fixed a bug in the copy constructor - 'm' was not copied.

  Version 1.22 (23 Mar 2002)
  changed creation and init functions; number of inputs and outputs are no longer
  specified directly, but are taken from the generator matrix size. Also changed the
  serialization functions; now these do not read/write k & n but only the generator.

  Version 1.30 (27 Mar 2002)
  changed descriptive output function to conform with fsm 1.40.

  Version 1.31 (13 Apr 2002)
  modified init() and constructor to take generator matrix by reference rather than
  directly.

  Version 1.40 (8 Jan 2006)
  modified class to conform with fsm 1.50.

  Version 1.50 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.51 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

class nrcc : public fsm {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new nrcc; };
   int k, n;                          // number of input and output bits, respectively
   int K;                                // number of memory elements (constraint length)
   int m;                                // memory order (longest input register)
   libbase::bitfield *reg;                    // shift registers (one for each input bit)
   libbase::matrix<libbase::bitfield> gen;      // generator sequence
protected:
   void init(const libbase::matrix<libbase::bitfield>& generator);
   nrcc();
public:
   // class management (construction/destruction)
   nrcc(const libbase::matrix<libbase::bitfield>& generator);
   nrcc(const nrcc& x);         // copy constructor
   ~nrcc();
   
   // class management (cloning/naming)
   nrcc *clone() const { return new nrcc(*this); };             // cloning operation
   const char* name() const { return shelper.name(); };

   // FSM operations (reset/advance/step/state)
   void reset(int state=0);     // resets the FSM to a specified state
   void resetcircular(int zerostate, int n); // resets the FSM, given the zero-state solution and the number of time-steps
   void resetcircular();
   void advance(int& input);  // feeds the specified input and advances the state
   int output(int& input);         // computes the output for the given input and the present state
   int step(int& input);           // feeds the specified input and returns the corresponding output
   int state() const;              // returns the current state

   // informative functions
   int num_states() const { return 1<<K; };     // returns the number of defined states
   int num_inputs() const { return 1<<k; };     // returns the number of valid inputs
   int num_outputs() const { return 1<<n; };    // returns the number of valid outputs
   int mem_order() const { return m; };    // memory order (length of tail)

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

