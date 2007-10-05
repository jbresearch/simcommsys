#include "montecarlo.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include "cmpi.h"
#include <iostream.h>

const vcs montecarlo_version("Monte Carlo Estimator module (montecarlo)", 1.11);

const int montecarlo::min_passes = 128;

// Static objects

bool montecarlo::init = false;
experiment *montecarlo::system;


// Non-static objects

montecarlo::montecarlo(experiment *system)
   {
   if(init)
      {
      cerr << "FATAL ERROR (montecarlo): cannot initialise more than one object.\n";
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

montecarlo::~montecarlo()
   {
   init = false;
   }
   
void montecarlo::set_confidence(const double confidence)
   {
   if(confidence <= 0.5 || confidence >= 1.0)
      {
      cerr << "ERROR: trying to set invalid confidence level of " << confidence << "\n";
      return;
      }
   secant Qinv(Q);	// init Qinv as the inverse of Q(), using the secant method
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

void montecarlo::child_init(void)
   {
   system->seed(cmpi::rank());
   double x;
   cmpi::_receive(x);
   system->set(x);
   }

void montecarlo::child_work(void)
   {
   const int count = system->count();
   vector<double> est(count);
   int sc = 0;
   system->sample(est, sc);
   cmpi::_send(est);
   cmpi::_send(sc);
   }

void montecarlo::estimate(vector<double>& result, vector<double>& tolerance)
   {
   // simulation timer
   timer tim;

   // MPI objects
   int mpi_outstanding = 0;
   int nprocs = 0;
   cmpi *mpi = NULL;

   // set stderr to 3 significant figures
   int prec = cerr.precision(3);

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

   // Start the timer
   tim.start();

   // Seed the experiment
   if(cmpi::enabled())
      {
      mpi = new cmpi;
      nprocs = mpi->size();
      for(int i=0; i<nprocs; i++)
         {
         mpi->call(i, &child_init);
         mpi->send(i, system->get());
         mpi->call(i, &child_work);
         }
      mpi_outstanding = nprocs;
      }
   else
      system->seed(0);

   // Repeat the experiment until we get the accuracy we need
   int passes = 0;
   while(!accuracy_reached || mpi_outstanding > 0)
      {
      // repeat the experiment
      if(cmpi::enabled())
         {
         int rank, sc;
         mpi->receive(rank, est);
         mpi->receive(rank, sc);
         samplecount += sc;
         mpi_outstanding--;
         if(!accuracy_reached)
            {
            mpi->call(rank, &child_work);
            mpi_outstanding++;
            }
         }
      else
         system->sample(est, samplecount);
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
      // check if we are ready
      if(passes >= min_passes && acc <= accuracy && acc != 0)
         accuracy_reached = true;
      // print something to inform the user of our progress
      cerr << "MC:\t" << passes << "\t" << (acc<1 ? 100*acc : 99) << "%\t[" << result(0) << "] \r";
      }

   // Kill children
   if(cmpi::enabled())
      delete mpi;

   // Stop the timer, divide by the number of samples, and print out
   tim.stop();
   cerr << "\nEstimate worked out in " << tim;
   tim.divide(passes);
   cerr << " (at " << tim << " per sample)\n";

   // reset stderr precision to what it was before
   cerr.precision(prec);
   }
