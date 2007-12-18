/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "montecarlo.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "truerand.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using std::cerr;
using libbase::trace;
using libbase::vector;

const int montecarlo::min_samples = 128;

// worker processes

void montecarlo::slave_getcode(void)
   {
   std::string systemstring;
   if(!receive(systemstring))
      exit(1);
   delete system;
   std::istringstream is(systemstring);
   is >> system;
   
   cerr << "Date: " << libbase::timer::date() << "\n";
   cerr << system->description() << "\n"; 
   }

void montecarlo::slave_getsnr(void)
   {
   libbase::truerand r;
   const libbase::int32u seed = r.ival(1<<16);
   system->seed(seed);
   double x;
   if(!receive(x))
      exit(1);
   system->set(x);

   cerr << "Date: " << libbase::timer::date() << "\n";
   cerr << "Seed: " << seed << "\n";
   cerr << "Simulating system at Eb/No = " << system->get() << "\n";
   }

void montecarlo::slave_work(void)
   {
   const int count = system->count();
   vector<double> est(count);
   system->sample(est);
   if(!send(est))
      exit(1);

   //// iterate for 500ms, which is a good compromise between efficiency and usability
   //int passes=0;
   //libbase::timer t;
   //while(t.elapsed() < 0.5)
   //   {
   //   cycleonce(result);   // will update result
   //   passes++;
   //   samplecount++;
   //   }
   //t.stop();   // to avoid expiry
   //// update result
   //result /= double(passes);
   }

// helper functions

void montecarlo::createfunctors(void)
   {
   fgetcode = new libbase::specificfunctor<montecarlo>(this, &libcomm::montecarlo::slave_getcode);
   fgetsnr = new libbase::specificfunctor<montecarlo>(this, &libcomm::montecarlo::slave_getsnr);
   fwork = new libbase::specificfunctor<montecarlo>(this, &libcomm::montecarlo::slave_work);
   // register functions
   fregister("slave_getcode", fgetcode);
   fregister("slave_getsnr", fgetsnr);
   fregister("slave_work", fwork);
   }

void montecarlo::destroyfunctors(void)
   {
   delete fgetcode;
   delete fgetsnr;
   delete fwork;
   }

// overrideable user-interface functions

void montecarlo::display(const int pass, const double cur_accuracy, const double cur_mean)
   {
   static libbase::timer tupdate;
   if(tupdate.elapsed() > 0.5)
      {
      std::clog << "Timer: " << t << ", " << getnumslaves() << " clients, " \
      << getcputime()/t.elapsed() << "x speedup, pass " << pass << ", " \
      << "[" << cur_mean << " +/- " << cur_accuracy << "%] \r" << std::flush;
      tupdate.start();
      }
   }

// constructor/destructor

montecarlo::montecarlo(experiment *system)
   {
   createfunctors();
   initialise(system);
   }

montecarlo::montecarlo()
   {
   createfunctors();
   init = false;
   system = NULL;
   }

montecarlo::~montecarlo()
   {
   if(init)
      finalise();
   destroyfunctors();
   }

// simulation initialization/finalization

void montecarlo::initialise(experiment *system)
   {
   if(init)
      {
      cerr << "ERROR (montecarlo): object already initialized.\n";
      exit(1);
      }
   init = true;
   // bind sub-components
   montecarlo::system = system;
   // reset the count of samples
   samplecount = 0;
   // set default parameter settings
   set_confidence(0.95);
   set_accuracy(0.10);
   }

void montecarlo::finalise()
   {
   init = false;
   }

// simulation parameters

void montecarlo::set_confidence(const double confidence)
   {
   if(confidence <= 0.5 || confidence >= 1.0)
      {
      cerr << "ERROR: trying to set invalid confidence level of " << confidence << "\n";
      return;
      }
   libbase::secant Qinv(libbase::Q);  // init Qinv as the inverse of Q(), using the secant method
   cfactor = Qinv((1.0-confidence)/2.0);
   }

void montecarlo::set_accuracy(const double accuracy)
   {
   if(accuracy <= 0 || accuracy >= 1.0)
      {
      cerr << "ERROR: trying to set invalid accuracy level of " << accuracy << "\n";
      return;
      }
   montecarlo::accuracy = accuracy;
   }

// main process

/*!
   \brief Update overall estimate, given last sample
   \param[in,out] passes      Number of passes, to be updated
   \param[in,out] result      Results to be updated
   \param[in,out] tolerance   Corresponding result accuracy to be updated
   \param[in,out] sum         Sum of results (to be updated)
   \param[in,out] sumsq       Sum of squares of results (to be updated)
   \param[in,out] nonzero     Count of passes where corresponding result was non-zero
   \param[in]     est         Last sample
   \return  Accuracy reached (worst accuracy over result set)

   \note There is a "bail-out" facility included, whereby we won't take a particular
         result's tolerance into account if the ratio of the number of non-zero estimates
         is less than 1/max_passes. This bail-out is avoided by setting max_passes = 0.
*/
double montecarlo::updateresults(int &passes, vector<double>& result, vector<double>& tolerance, vector<double>& sum, vector<double>& sumsq, const vector<double>& est)
   {
   const int count = result.size();
   // update the number of passes
   passes++;
   // for each result:
   double acc = 0;
   for(int i=0; i<count; i++)
      {
      // update the running totals
      sum(i) += est(i);
      sumsq(i) += est(i)*est(i);
      // work mean and sd
      double mean = sum(i)/double(passes);
      double sd = sqrt((sumsq(i)/double(passes) - mean*mean)/double(passes-1));
      // update results
      result(i) = mean;
      if(mean > 0)
         {
         tolerance(i) = cfactor*sd/mean;
         if(tolerance(i) > acc)
            acc = tolerance(i);
         }
      }
   return acc;
   }

/*!
   \brief Initialize any new slaves
   \param   systemstring   Serialized system description

   If there are any slaves in the NEW state, initialize them by sending the system
   being simulated and the current simulation parameters.
*/
void montecarlo::initnewslaves(std::string systemstring)
   {
   while(slave *s = newslave())
      {
      trace << "DEBUG (estimate): New slave found (" << s << "), initializing.\n";
      if(!call(s, "slave_getcode"))
         continue;
      if(!send(s, systemstring))
         continue;
      if(!call(s, "slave_getsnr"))
         continue;
      if(!send(s, system->get()))
         continue;
      trace << "DEBUG (estimate): Slave (" << s << ") initialized ok.\n";
      }
   }

/*!
   \brief Get idle slaves to work if we're not yet done
   \param   accuracy_reached  Flag to indicate if we have already reached required accuracy

   If there are any slaves in the IDLE state, ask them to start working. If the target
   accuracy is not yet reached, we ask *all* IDLE slaves to work. Otherwise, we only ask
   the necessary number so that when they're done we will have reached the minimum samples
   limit. This avoids having to wait for more slaves to finish once all necessary conditions
   (ie. target accuracy and minimum number of samples) are met.
*/
void montecarlo::workidleslaves(bool accuracy_reached)
   {
   for(slave *s; (!accuracy_reached || samplecount+workingslaves() < min_samples) && (s = idleslave()); )
      {
      trace << "DEBUG (estimate): Idle slave found (" << s << "), assigning work.\n";
      if(!call(s, "slave_work"))
         continue;
      trace << "DEBUG (estimate): Slave (" << s << ") work assigned ok.\n";
      }
   }

/*!
   \brief Simulate the system until convergence to given accuracy & confidence,
          and return estimated results
   \param[out] result      Vector of results
   \param[out] tolerance   Vector of corresponding result accuracy (at given confidence level)
*/
void montecarlo::estimate(vector<double>& result, vector<double>& tolerance)
   {
   const int prec = std::clog.precision(3);
   t.start();

   // Initialise space for results
   const int count = system->count();
   result.init(count);
   tolerance.init(count);

   // Running values
   vector<double> est(count), sum(count), sumsq(count);
   bool accuracy_reached = false;

   // Initialise running values
   samplecount = 0;
   sum = 0;
   sumsq = 0;

   // Seed the experiment
   std::string systemstring;
   if(isenabled())
      {
      std::ostringstream os;
      os << system;
      systemstring = os.str();
      resetslaves();
      }
   else
      system->seed(0);

   // Repeat the experiment until all the following are true:
   // 1) We have the accuracy we need
   // 2) We have enough samples for the accuracy to be meaningful
   // 3) No slaves are still working
   // An interrupt from the user overrides everything...
   int passes = 0;
   while(!accuracy_reached || samplecount < min_samples || (isenabled() && anyoneworking()))
      {
      bool results_available = false;
      // repeat the experiment
      if(isenabled())
         {
         // first initialize any new slaves
         initnewslaves(systemstring);
         // get idle slaves to work if we're not yet done
         workidleslaves(accuracy_reached);
         // wait for results, but not indefinitely - this allows user to break
         trace << "DEBUG (estimate): Waiting for event.\n";
         waitforevent(true, 0.5);
         // get set of results from the pending slave, if any
         // *** we assume here that waitforevent() only returns one pending slave at most ***
         if(slave *s = pendingslave())
            {
            trace << "DEBUG (estimate): Pending event from slave (" << s << "), trying to read.\n";
            if(receive(s, est))
               {
               trace << "DEBUG (estimate): Read from slave (" << s << ") succeeded.\n";
               samplecount++;
               updatecputime(s);
               results_available = true;
               }
            }
         }
      else
         {
         system->sample(est);
         samplecount++;
         results_available = true;
         }
      // if we did get any results, update the statistics
      if(results_available)
         {
         double acc = updateresults(passes, result, tolerance, sum, sumsq, est);
         // check if we have reached the required accuracy
         if(acc <= accuracy && acc != 0)
            accuracy_reached = true;
         // print something to inform the user of our progress
         display(passes, (acc<1 ? 100*acc : 99), result(0));
         }
      // consider our work done if the user has interrupted the processing (this overrides everything)
      if(interrupt())
         break;
      }

   t.stop();
   std::clog.precision(prec);
   }

}; // end namespace
