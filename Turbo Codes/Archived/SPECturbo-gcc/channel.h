#ifndef __channel_h
#define __channel_h

#include "config.h"
#include "vcs.h"
#include "sigspace.h"

extern const vcs channel_version;

class channel {
public:
   virtual void seed(const int32u s) = 0;
   virtual void set_eb(const double Eb) = 0;
   virtual void set_snr(const double snr_db) = 0;
   virtual double get_snr() const = 0;
   virtual sigspace corrupt(const sigspace& s) = 0;
   virtual double pdf(const sigspace& tx, const sigspace& rx) const = 0;
};

#endif

