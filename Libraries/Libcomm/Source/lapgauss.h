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
  Version 1.00 (10 Aug 2006)
  Initial version - implementation of the additive Laplacian-Gaussian channel model.

  Version 1.10 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
*/

namespace libcomm {

class lapgauss : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new lapgauss; };
   libbase::randgen		r;
   double		sigma, Eb, No, snr_db;
public:
   lapgauss();

   channel *clone() const { return new lapgauss(*this); };		// cloning operation
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

