#ifndef __awgn_h
#define __awgn_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

/*
  Version 1.00 (12-16 Oct 2007)
  Initial version; implementation of a binary substitution, insertion, and deletion channel.
  * TODO: this class is still unfinished, and only implements the BSC channel right now
*/

namespace libcomm {

class bsid : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new bsid; };
   // channel paremeters
   double   Ps, Pd, Pi;       // specific parameters
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
public:
   // object handling
   bsid();
   channel *clone() const { return new bsid(*this); };
   const char* name() const { return shelper.name(); };

   // channel functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;

   // description output
   std::string description() const;
   // object serialization
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

