#ifndef __fsm_h
#define __fsm_h

#include "config.h"
#include "vcs.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   .
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.10 (4 Nov 2001)
  added a virtual function which outputs details on the finite state machine (this was 
  only done before in a non-standard print routine). Added a stream << operator too.

  Version 1.20 (28 Feb 2002)
  added serialization facility

  Version 1.21 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.22 (11 Mar 2002)
  changed the definition of the cloning operation to be a const member.

  Version 1.30 (11 Mar 2002)
  changed the stream << and >> functions to conform with the new serializer protocol,
  as defined in serializer 1.10. The stream << output function first writes the name
  of the derived class, then calls its serialize() to output the data. The name is
  obtained from the virtual name() function. The stream >> input function first gets
  the name from the stream, then (via serialize::call) creates a new object of the
  appropriate type and calls its serialize() function to get the relevant data. Also,
  changed the definition of stream << output to take the pointer to the fsm class
  directly, not by reference.

  Version 1.40 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

  Version 1.50 (8 Jan 2006)
  updated class to allow consistent support for circular trellis encoding (tail-biting)
  and also for the use of rate m/(m+1) LFSR codes. In practice this has been achieved
  by creating a generalized convolutional code class that uses state-space techniques
  to represent the system - the required codes, such as rate m/(m+1) LFSR including
  the DVB duo-binary codes can be easily derived. The following virtual functions
  were added to allow this:
  * advance(input) - this performs the state-change without also computing the output;
    it is provided as a faster version of step(), since the output doesn't need to be
    computed in the first iteration.
  * output(input) - this is added for completeness, and calculates the output, given
    the present state and input.
  * resetcircular(zerostate, N) - this performs the initial state computation (and
    setting) for circular encoding. It is assumed that this will be called after a
    sequence of the form [reset(); loop step()/advance()] which the calling class
    uses to determine the zero-state solution. N is the number of time-steps involved.
  * resetcircular() - is a convenient form of the above function, where the fsm-derived
    class must keep track of the number of time-steps since the last reset operation
    as well as the final state value. The calling class must ensure that this is
    consistent with the requirements - that is, the initial reset must be to state zero
    and the input sequence given since the last reset must be the same as the one that
    will be used now.
  It was elected to make all the above functions pure virtual - this requires that all
  derived classes must be updates accordingly; it is considered advantageous because
  there are just two derived classes which can therefore be easily updated. This avoids
  adding dependance complexity in the fsm class (there is less risk of unexpected
  behaviour).

  Version 1.60 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.70 (3-5 Dec 2007)
  * Updated output() so that the input value is a const and the function is also a const
  * provided a default implementation of step() using output() and advance()
  * cleaned up order of members and documentation
  * removed friend status of stream output operators

  Version 1.71 (13 Dec 2007)
  * modified parameter type for output from "const int&" to "int"
*/

class fsm {
   static const libbase::vcs version;
public:
   static const int tail;                 // a special input to use when tailing out
   
   // class management (construction/cloning/naming)
   virtual ~fsm() {};                     // virtual destructor
   virtual fsm *clone() const = 0;        // cloning operation
   virtual const char* name() const = 0;  // derived object's name

   // FSM state operations (getting and resetting)
   virtual int state() const = 0;         // returns the current state
   virtual void reset(int state=0) = 0;   // reset to a specified state
   virtual void resetcircular(int zerostate, int n) = 0; // resets, given zero-state solution and number of time-steps
   virtual void resetcircular() = 0;      // as above, assuming we have just run through the zero-state zero-input
   // FSM operations (advance/output/step)
   virtual int output(int input) const = 0; // computes the output for the given input and the present state
   virtual void advance(int& input) = 0;  // feeds the specified input and advances the state
   virtual int step(int& input);          // feeds the specified input and returns the corresponding output

   // informative functions
   virtual int mem_order() const = 0;     // memory order (length of tail)
   virtual int num_states() const = 0;    // returns the number of defined states
   virtual int num_inputs() const = 0;    // returns the number of valid inputs
   virtual int num_outputs() const = 0;   // returns the number of valid outputs

   // description output
   virtual std::string description() const = 0;
   // object serialization
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   virtual std::istream& serialize(std::istream& sin) = 0;
};

// stream output operators
std::ostream& operator<<(std::ostream& sout, const fsm* x);
std::istream& operator>>(std::istream& sin, fsm*& x);

}; // end namespace

#endif

