#ifndef __laplacian_h
#define __laplacian_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

/*
  Version 1.00 (15 Apr 2001)
  First version - note that as with the Gaussian channel, the distribution
  has zero mean even in this case.

  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

  Version 1.10 (13 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with channel 1.10), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too). Also made the channel object a public base class, rather than a
  virtual public one, since this was affecting the transfer of virtual functions within
  the class (causing access violations). Also moved most functions into the implementation
  file rather than here.

  Version 1.20 (27 Mar 2002)
  changed descriptive output function to conform with channel 1.30.

  Version 1.30 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.

  Version 1.40 (16 Oct 2007)
  changed class to conform with channel 1.50.

  Version 1.41 (16 Oct 2007)
  changed class to conform with channel 1.51.

  Version 1.42 (17 Oct 2007)
  changed class to conform with channel 1.52.

  Version 1.43 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

class laplacian : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new laplacian; };
   // channel paremeters
   double   lambda;
private:
   // internal helper functions
   double f(const double x) const { return 1/(2*lambda) * exp(-fabs(x)/lambda); };
   double Finv(const double y) const { return (y < 0.5) ? lambda*log(2*y) : -lambda*log(2*(1-y)); };
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // object handling
   laplacian *clone() const { return new laplacian(*this); };
   const char* name() const { return shelper.name(); };

   // description output
   std::string description() const;
};

}; // end namespace

#endif

