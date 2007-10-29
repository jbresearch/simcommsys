#ifndef __watermarkcode_h
#define __watermarkcode_h

#include "config.h"
#include "vcs.h"

#include "modulator.h"
#include "mpsk.h"
#include "fba.h"
#include "logrealfast.h"
#include "bsid.h"

#include "bitfield.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

/*
  Version 1.00 (1-12 Oct 2007)
  initial version; implements Watermark Codes as described by Davey in "Reliable
  Communication over Channels with Insertions, Deletions, and Substitutions", Trans. IT,
  Feb 2001.
  
  Version 1.10 (25-29 Oct 2007)
  * made class a 'modulator' rather than a 'modulator' as this better reflects its position within
    the communication model's stack
  * removed 'N' from a code parameter - this is simply the size of the block currently being demodulated
  * removed 'const' restriction on modulate and demodulate vector functions, as in modulator 1.50
  * added private inheritance from bsid channel (to access the transmit and receive functions there)
  * updated clone() to return type 'watermarkcode' instead of 'modulator'; this avoids a derivation ambiguity
    problem introduced with the inheritance from bsid, which is a channel, not a modulator. [cf. Stroustrup 15.6.2]
  * added assertions during initialization
  * started implementations of P(), Q() and demodulate()
    TODO: finish demodulation and test
*/

namespace libcomm {

template <class real> class watermarkcode : public mpsk, private bsid, private fba<libbase::logrealfast> {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new watermarkcode<real>; };
private:
   // user-defined parameters
   int      n, k, s;    // code parameters: #bits in sparse (output) symbol, message (input) symbol; generator seed
   int      I, xmax;    // decoder parameters
   double   Ps, Pd, Pi; // channel parameters
   // computed parameters
   double   f, Pf, Pt, alphaI;
   // internally-used objects
   libbase::randgen r;        // watermark sequence generator
   libbase::vector<int> lut;  // sparsifier LUT
   // LUT creation
   int fill(int i, libbase::bitfield suffix, int weight);
   // implementations of channel-specific metrics for fba
   double P(const int a, const int b);
   double Q(const int a, const int b, const int i, const int s);
   // modulation/demodulation - atomic operations (private as these should never be used)
   const sigspace modulate(const int index) const { return sigspace(0,0); };
   const int demodulate(const sigspace& signal) const { return 0; };
protected:
   void init();
   void free();
   watermarkcode();
public:
   watermarkcode(const int n, const int k, const int s, \
      const int I, const int xmax, const double Ps, const double Pd, const double Pi);
   ~watermarkcode() { free(); };

   watermarkcode *clone() const { return new watermarkcode(*this); };		// cloning operation
   const char* name() const { return shelper.name(); };

   // modulation/demodulation - atomic operations
   // const sigspace modulate(const int index) const { return sigspace(0,0); };
   // const int demodulate(const sigspace& signal) const { return 0; };

   // modulation/demodulation - vector operations
   //    N - the number of possible values of each encoded element
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx);
   void demodulate(const channel& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable);

   // information functions
   int num_symbols() const { return 1<<k; };
   double energy() const { return 1<<n; };  // average energy per symbol

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

