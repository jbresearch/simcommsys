#ifndef __experiment_h
#define __experiment_h

#include "config.h"
#include "vector.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Generic experiment.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (2 Sep 1999)
   added a hook for clients to know the number of frames simulated in a particular run.

   \version 1.11 (26 Oct 2001)
   added a virtual destroy function (see interleaver.h)

   \version 1.12 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.

   \version 1.20 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.

   \version 1.30 (24 Apr 2007)
   - added serialization facility requirement, to facilitate passing the experiment
    over the network pipe for masterslave implementation.

   \version 1.40 (18 Dec 2007)
   - Modified definition of sample() so that only a *single* sample is performed.
     This is essential so that the moments of the results are meaningful and can
     be used to determine convergence/tolerance. Any multiple-sampling should be
     done elsewhere (e.g. in montecarlo::slave_work).
*/

class experiment {
public:
   virtual ~experiment() {};                 // virtual destructor
   virtual experiment *clone() const = 0;    // cloning operation
   virtual const char* name() const = 0;     // derived object's name
   
   virtual int count() const = 0;
   virtual void seed(int s) = 0;
   virtual void set(double x) = 0;
   virtual double get() = 0;
   /*!
   \brief Perform the experiment and return a single sample
   \param[out] result   Vector containing the set of results for the experiment
   */
   virtual void sample(libbase::vector<double>& result) = 0;
   
   // description output
   virtual std::string description() const = 0;
   // object serialization - saving
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   friend std::ostream& operator<<(std::ostream& sout, const experiment* x);
   // object serialization - loading
   virtual std::istream& serialize(std::istream& sin) = 0;
   friend std::istream& operator>>(std::istream& sin, experiment*& x);
};

}; // end namespace

#endif
