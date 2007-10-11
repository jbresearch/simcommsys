#ifndef __watermarkcode_h
#define __watermarkcode_h

#include "config.h"
#include "vcs.h"

#include "codec.h"
#include "logrealfast.h"
#include "fba.h"
#include "bitfield.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

/*
  Version 1.00 (1-9 Oct 2007)
  initial version; implements Watermark Codes as described by Davey in "Reliable
  Communication over Channels with Insertions, Deletions, and Substitutions", Trans. IT,
  Feb 2001.
*/

namespace libcomm {

template <class real> class watermarkcode : public codec, private fba<libbase::logrealfast> {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new watermarkcode<real>; };
private:
   // code parameters
   int   N, n, k, s;
   // watermark sequence generator
   libbase::randgen r;
   // sparsifier LUT
   libbase::vector<int> lut;
   // LUT creation
   int fill(int i, libbase::bitfield suffix, int weight);
   // implementations of channel-specific metrics for fba
   double P(const int a, const int b);
   double Q(const int a, const int b, const int i, const int s);
protected:
   void init();
   void free();
   watermarkcode();
public:
   watermarkcode(const int N, const int n, const int k, const int s);
   ~watermarkcode() { free(); };

   codec *clone() const { return new watermarkcode(*this); };		// cloning operation
   const char* name() const { return shelper.name(); };

   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void translate(const libbase::matrix<double>& ptable);
   void decode(libbase::vector<int>& decoded);
   void decode(libbase::matrix<double>& decoded);
   
   int block_size() const { return N; };
   int num_inputs() const { return 1<<k; };
   int num_outputs() const { return 1<<n; };
   int tail_length() const { return 0; };
   int num_iter() const { return 1; };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

