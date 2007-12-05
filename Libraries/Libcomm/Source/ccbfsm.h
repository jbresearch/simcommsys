#ifndef __ccbfsm_h
#define __ccbfsm_h

#include "vcs.h"
#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "vector.h"

/*
  Version 1.00 (5 Dec 2007)
  * initial version; implements common elements of a controller-canonical binary fsm
  * added determineinput() and determinefeedin() pure virtual functions
  * cleaned up order of members and documentation
*/

namespace libcomm {

class ccbfsm : public fsm {
   static const libbase::vcs version;
protected:
   int k, n;   // number of input and output bits, respectively
   int nu;     // number of memory elements (constraint length)
   int m;      // memory order (longest input register)
   libbase::vector<libbase::bitfield> reg;   // shift registers (one for each input bit)
   libbase::matrix<libbase::bitfield> gen;   // generator sequence
private:
   void init(const libbase::matrix<libbase::bitfield>& generator);
protected:
   virtual libbase::bitfield determineinput(const int input) const = 0;
   virtual libbase::bitfield determinefeedin(const int input) const = 0;
   ccbfsm();
public:
   // class management (construction/destruction)
   ccbfsm(const libbase::matrix<libbase::bitfield>& generator);
   ccbfsm(const ccbfsm& x);
   ~ccbfsm();
   
   // FSM state operations (getting and resetting)
   int state() const;                        // returns the current state
   void reset(int state=0);                  // reset to a specified state
   // FSM operations (advance/output/step)
   void advance(int& input);                 // feeds the specified input and advances the state

   // informative functions
   int mem_order() const { return m; };      // memory order (length of tail)
   int num_states() const { return 1<<nu; }; // returns the number of defined states
   int num_inputs() const { return 1<<k; };  // returns the number of valid inputs
   int num_outputs() const { return 1<<n; }; // returns the number of valid outputs

   // description output - common part only, must be preceded by specific name
   std::string description() const;
   // object serialization
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

