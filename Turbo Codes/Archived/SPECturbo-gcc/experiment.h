#ifndef __experiment_h
#define __experiment_h

#include "config.h"
#include "vcs.h"

#include "vector.h"

extern const vcs experiment_version;

/*
  Version 1.10 (2 Sep 1999)
  added a hook for clients to know the number of frames simulated in a particular run.
*/
class experiment {
public:
   virtual int count() const = 0;
   virtual void seed(int s) = 0;
   virtual void set(double x) = 0;
   virtual double get() = 0;
   virtual void sample(vector<double>& result, int& samplecount) = 0;
};

#endif
