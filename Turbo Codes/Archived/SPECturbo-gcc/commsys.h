#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "vcs.h"

#include "experiment.h"
#include "randgen.h"
#include "channel.h"
#include "codec.h"

extern const vcs commsys_version;

/*
  Version 1.00

  Version 1.10 (7 Jun 1999)
  modified the system to comply with codec 1.10.

  Version 1.20 (30 Jul 1999)
  added option to speed up Turbo decoding (by stopping when an iteration does not
  improve the error rate).

  Version 1.21 (26 Aug 1999)
  modified stopping criterion for samples such that sample granularity is just above 0.5s
  based on a timer rather than on the number of symbols transmitted

  Version 1.30 (2 Sep 1999)
  added a hook for clients to know the number of frames simulated in a particular run.
*/
class commsys : public virtual experiment {
   // bound objects:
   randgen  *src;
   channel  *chan;
   codec    *cdc;
   // working variables (data heap)
   bool fast;
   int	tau, n, m, K, k, iter;
   vector<int> source, encoded, decoded, last;
   vector<sigspace>  received;
private:
   void cycleonce(vector<double>& result);
public:
   commsys(randgen *src, channel *chan, codec *cdc, bool fast=false);
   ~commsys();
   int count() const { return 2*iter; };
   void seed(int s);
   void set(double x);
   double get();
   void sample(vector<double>& result, int& samplecount);
};

#endif
