/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "experiment.h"
#include <limits>

namespace libcomm {

// Normally-distributed sample experiment

void experiment_normal::derived_reset()
   {
   assert(count() > 0);
   // Initialise space for running values
   sum.init(count());
   sumsq.init(count());
   // Initialise running values
   sum = 0;
   sumsq = 0;
   }

void experiment_normal::derived_accumulate(const libbase::vector<double>& result)
   {
   assert(count() == result.size());
   assert(count() == sum.size());
   assert(count() == sumsq.size());
   // accumulate results
   libbase::vector<double> sample = result;
   sum += sample;
   sample.apply(square);
   sumsq += sample;
   }

void experiment_normal::accumulate_state(const libbase::vector<double>& state)
   {
   assert(count() == sum.size());
   assert(count() == sumsq.size());
   assert(2*count() == state.size());
   for(int i=0; i<count(); i++)
      {
      sum(i) += state(i);
      sumsq(i) += state(count()+i);
      }
   }

void experiment_normal::get_state(libbase::vector<double>& state) const
   {
   assert(count() == sum.size());
   assert(count() == sumsq.size());
   state.init(2*count());
   for(int i=0; i<count(); i++)
      {
      state(i) = sum(i);
      state(count()+i) = sumsq(i);
      }
   }

void experiment_normal::estimate(libbase::vector<double>& estimate, libbase::vector<double>& stderror) const
   {
   assert(count() == sum.size());
   assert(count() == sumsq.size());
   // estimate is the mean value
   assert(get_samplecount() > 0);
   estimate = sum/double(get_samplecount());
   // standard error is sigma/sqrt(n)
   stderror.init(count());
   if(get_samplecount() > 1)
      for(int i=0; i<count(); i++)
         stderror(i) = sqrt((sumsq(i)/double(get_samplecount()) - estimate(i)*estimate(i))/double(get_samplecount()-1));
   else
      stderror = std::numeric_limits<double>::max();
   }

// Experiment for estimation of a binomial proportion

void experiment_binomial::derived_reset()
   {
   assert(count() > 0);
   // Initialise space for running values
   sum.init(count());
   // Initialise running values
   sum = 0;
   }

void experiment_binomial::derived_accumulate(const libbase::vector<double>& result)
   {
   assert(count() == result.size());
   assert(count() == sum.size());
   // accumulate results
   sum += result;
   }

void experiment_binomial::accumulate_state(const libbase::vector<double>& state)
   {
   assert(count() == sum.size());
   assert(count() == state.size());
   // accumulate results from saved state
   sum += state;
   }

void experiment_binomial::get_state(libbase::vector<double>& state) const
   {
   assert(count() == sum.size());
   state = sum;
   }

void experiment_binomial::estimate(libbase::vector<double>& estimate, libbase::vector<double>& stderror) const
   {
   assert(count() == sum.size());
   // initialize space for results
   estimate.init(count());
   stderror.init(count());
   // compute results
   assert(get_samplecount() > 0);
   for(int i=0; i<count(); i++)
      {
      // estimate is the proportion
      estimate(i) = sum(i)/double(get_samplecount(i));
      // standard error is sqrt(p(1-p)/n)
      stderror(i) = sqrt( (estimate(i)*(1-estimate(i))) / double(get_samplecount(i)) );
      }
   }

}; // end namespace
