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

const libbase::vcs montecarlo::version("Monte Carlo Estimator module (montecarlo)", 1.31);

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
   system->seed(r.ival(1<<16));
   double x;
   if(!receive(x))
      exit(1);
   system->set(x);

   cerr << "Date: " << libbase::timer::date() << "\n";
   cerr << "Simulating system at Eb/No = " << system->get() << "\n";
   }

void montecarlo::slave_work(void)
   {
   const int count = system->count();
   vector<double> est(count);
   int sc = 0;
   system->sample(est, sc);
   if(!send(est) || !send(sc))
      exit(1);
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
   std::clog << "MC: " << t << " elapsed, " << getnumslaves() << " clients, " \
      << getcputime()/t.elapsed() << " speedup, pass " << pass << ", accuracy = " << cur_accuracy << "%, [" << cur_mean << "] \r" << std::flush;
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
   set_bailout(0);
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

void montecarlo::set_bailout(const int passes)
   {
   if(passes < 0)
      {
      cerr << "ERROR: trying to set invalid bailout level of 1 in " << passes << "\n";
      return;
      }
   montecarlo::max_passes = passes;
   }

// main process

void montecarlo::estimate(vector<double>& result, vector<double>& tolerance)
   {
   const int prec = std::clog.precision(3);
   t.start();

   // Running values
   const int count = system->count();
   vector<double> est(count), sum(count), sumsq(count);
   vector<int> nonzero(count);
   bool accuracy_reached = false;

   // Initialise running values
   for(int i=0; i<count; i++)
      sum(i) = sumsq(i) = nonzero(i) = 0;

   // Initialise results
   result.init(count);
   tolerance.init(count);
   samplecount = 0;

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
         // get idle slaves to work if we're not yet done
         for(slave *s; (!accuracy_reached || samplecount+workingslaves() < min_samples) && (s = idleslave()); )
            {
            trace << "DEBUG (estimate): Idle slave found (" << s << "), assigning work.\n";
            call(s, "slave_work");
            }
         // wait for results, but not indefinitely - this allows user to break
         trace << "DEBUG (estimate): Waiting for event.\n";
         waitforevent(true, 0.5);
         // get set of results from the pending slave, if any
         // *** we assume here that waitforevent() only returns one pending slave at most ***
         if(slave *s = pendingslave())
            {
            trace << "DEBUG (estimate): Pending event from slave (" << s << "), trying to read.\n";
            int sc;
            if(receive(s, est) && receive(s, sc))
               {
               trace << "DEBUG (estimate): Read from slave (" << s << ") succeeded.\n";
               samplecount += sc;
               updatecputime(s);
               results_available = true;
               }
            }
         }
      else
         {
         system->sample(est, samplecount);
         results_available = true;
         }
      // if we did get any results, update the statistics
      if(results_available)
         {
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
               if(est(i) > 0)
                  nonzero(i)++;
               // If we arrived here, nonzero(i) must be >0 because mean>0.
               // We won't take this tolerance into account if the ratio of the number of
               // non-zero estimates is less than 1/max_passes.
               // We set max_passes = 0 to indicate that we won't use this bailout facility
               if(tolerance(i) > acc && (max_passes==0 || passes < nonzero(i)*max_passes))
                  acc = tolerance(i);
               }
            }
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
