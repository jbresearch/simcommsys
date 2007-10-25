#ifndef __lut_modulator_h
#define __lut_modulator_h
      
#include "modulator.h"

/*
  Version 1.00 (25 Oct 2007)
  * initial version - contains LUT modulator implementation from modulator.h 1.41
  * refactored - renamed 'map' to 'lut' to better reflect the contents of the variable
  * removed 'const' restriction on modulate and demodulate vector functions, as in modulator 1.50
*/

namespace libcomm {

class lut_modulator : public modulator {
   static const libbase::vcs version;
protected:
   libbase::vector<sigspace> lut;   // Array of modulation symbols
public:
   virtual ~lut_modulator() {};     // virtual destructor

   // modulation/demodulation - atomic operations
   const sigspace modulate(const int index) const { return lut(index); };
   const int demodulate(const sigspace& signal) const;

   // modulation/demodulation - vector operations
   //    N - the number of possible values of each encoded element
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx);
   void demodulate(const channel& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable);

   // information functions
   int num_symbols() const { return lut.size(); };
   double energy() const;  // average energy per symbol
};

}; // end namespace

#endif
