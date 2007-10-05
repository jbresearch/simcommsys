#ifndef __puncture_stipple_h
#define __puncture_stipple_h

#include "config.h"
#include "vcs.h"

#include "puncture.h"

extern const vcs puncture_stipple_version;

/*
  Version 1.00 (7 Jun 1999)
  initial version, implementing an odd/even punctured system.
*/
class puncture_stipple : public virtual puncture {
public:
   puncture_stipple(const int tau, const int s);
   ~puncture_stipple();

   void print(ostream& s) const;
};

#endif
