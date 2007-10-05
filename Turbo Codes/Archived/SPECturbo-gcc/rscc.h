#ifndef __rscc_h
#define __rscc_h

#include "vcs.h"
#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"

extern const vcs rscc_version;

class rscc : public virtual fsm {
   int k, n;			// number of input and output bits, respectively
   int K;			// number of memory elements (constraint length)
   int m;			// memory order (longest input register)
   bitfield *reg;		// shift registers (one for each input bit)
   matrix<bitfield> gen;	// generator sequence
public:
   rscc(const int k, const int n, const matrix<bitfield> generator);
   rscc(const rscc& x);		// copy constructor
   ~rscc();
   
   fsm *clone();		// cloning operation

   void reset(int state=0);	// resets the FSM to a specified state
   int step(int& input);	// feeds the specified input and returns the corresponding output
   int state() const;		// returns the current state
   int num_states() const;	// returns the number of defined states
   int num_inputs() const;	// returns the number of valid inputs
   int num_outputs() const;	// returns the number of valid outputs
   int mem_order() const;	// memory order (length of tail)

   void print(ostream& s) const;
};

inline int rscc::num_states() const
   {
   return 1<<K;
   }

inline int rscc::num_inputs() const
   {
   return 1<<k;
   }

inline int rscc::num_outputs() const
   {
   return 1<<n;
   }

inline int rscc::mem_order() const
   {
   return m;
   }

#endif

