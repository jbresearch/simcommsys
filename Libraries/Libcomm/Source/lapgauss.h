#ifndef __awgn_h
#define __awgn_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

/*
  Version 1.00 (10 Aug 2006)
  Initial version - implementation of the additive Laplacian-Gaussian channel model.

  Version 1.10 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.

  Version 1.20 (16 Oct 2007)
  * changed class to conform with channel 1.50.
  * TODO: this class is still unfinished, and only implements the plain Gaussian channel right now

  Version 1.21 (17 Oct 2007)
  changed class to conform with channel 1.52.
*/

namespace libcomm {

class lapgauss : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new lapgauss; };
   // channel paremeters
   double		sigma;
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // object handling
   lapgauss();
   channel *clone() const { return new lapgauss(*this); };
   const char* name() const { return shelper.name(); };

   // description output
   std::string description() const;
   // object serialization
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

