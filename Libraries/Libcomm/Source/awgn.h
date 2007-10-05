#ifndef __awgn_h
#define __awgn_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

/*
  Version 1.10 (15 Apr 1999)
  Changed the definition of set_snr to avoid using the pow() function.
  This was causing an unexplained SEGV with optimised code

  Version 1.11 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

  Version 1.20 (13 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with channel 1.10), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too). Also made the channel object a public base class, rather than a
  virtual public one, since this was affecting the transfer of virtual functions within
  the class (causing access violations). Also moved most functions into the implementation
  file rather than here.

  Version 1.30 (27 Mar 2002)
  changed descriptive output function to conform with channel 1.30.

  Version 1.40 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
*/

namespace libcomm {

class awgn : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new awgn; };
   libbase::randgen		r;
   double		sigma, Eb, No, snr_db;
public:
   awgn();

   channel *clone() const { return new awgn(*this); };		// cloning operation
   const char* name() const { return shelper.name(); };

   void seed(const libbase::int32u s);
   void set_eb(const double Eb);
   void set_snr(const double snr_db);
   double get_snr() const { return snr_db; };
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

