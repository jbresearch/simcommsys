#ifndef __modulator_h
#define __modulator_h
      
#include "config.h"
#include "vcs.h"
#include "sigspace.h"
#include "itfunc.h"
#include <iostream.h>

extern const vcs modulator_version;

class modulator {
protected:
   int		M;	// Number of modulation symbols
   sigspace	*s;	// Array of modulation symbols
public:
   const sigspace& operator[](const int index) const;	// Modulation function
   const int operator[](const sigspace& signal) const;	// Demodulation function
   const int num_symbols() const;	// Returns the number of symbols
   const double energy() const;         // Returns the average energy per symbol
   const double bit_energy() const;     // Returns the average energy per bit
};

inline const int modulator::num_symbols() const
   {
   return M;
   }

inline const double modulator::bit_energy() const
   {
   return energy()/log2(M);
   }
   
#endif

