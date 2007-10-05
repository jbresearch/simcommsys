#ifndef __montecarlo_h
#define __montecarlo_h

#include "config.h"
#include "vcs.h"

#include "experiment.h"

extern const vcs montecarlo_version;

/*
  Version 1.10 (2 Sep 1999)
  added a hook for clients to know the number of frames simulated in a particular run.

  Version 1.11 (9 Sep 1999)
  changed min_passes to 128, since we observed that with 30 samples only, the distribution
  is still quite skewed.
*/
class montecarlo {
   // bound objects
   static bool init;
   static experiment *system;
   // internal variables
   static const int	min_passes;	// minimum number of passes to assume gaussian distribution
   int      max_passes; // max # of passes (with 1 non-zero result) affecting acceptance
   double   cfactor;	// factor dependent on confidence level
   double   accuracy;	// accuracy level required
   int      samplecount;      // number of samples taken to produce the result (updated by experiment module to allow for dynamic sampling)
   // MPI child process
   static void child_init(void);
   static void child_work(void);
public:
   montecarlo(experiment *system);
   ~montecarlo();
   
   void set_confidence(const double confidence);   // say, 0.95 => 95% probability
   void set_accuracy(const double accuracy);	   // say, 0.10 => 10% of mean
   void set_bailout(const int passes);             // say, 1000 => at least 1 in 1000 non-zero estimates (0 to disable)
   
   int get_samplecount() { return samplecount; };  // returns the number of samples taken to produce the result
   void estimate(vector<double>& result, vector<double>& tolerance);		// get an estimate with given accuracy & confidence
};

#endif
