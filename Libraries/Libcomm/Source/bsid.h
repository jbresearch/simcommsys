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
  Version 1.00 (12-16 Oct 2007)
  Initial version; implementation of a binary substitution, insertion, and deletion channel.
*/

namespace libcomm {

class bsid : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new bsid; };
   // objects used by the channel
   libbase::randgen  r;
   // channel paremeters
   double   Ps, Pd, Pi;       // specific parameters
   double   Eb, No, snr_db;   // base class interface parameters
private:
   // internal helper functions
   void compute_parameters();
public:
   bsid();

   channel *clone() const { return new bsid(*this); };		// cloning operation
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

