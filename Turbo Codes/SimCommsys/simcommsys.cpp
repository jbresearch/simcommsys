#include "randgen.h"
#include "serializer_libcomm.h"
#include "commsys.h"
#include "montecarlo.h"
#include "masterslave.h"
#include "timer.h"

#include <math.h>
#include <string.h>
#include <iostream>
#include <iomanip>

/*!
   \file
   \brief   Simulation of Communication Systems
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

using std::cout;
using std::cerr;
using std::setprecision;
using std::flush;

class mymontecarlo : public libcomm::montecarlo {
public:
   // make interrupt function public to allow use by main program
   bool interrupt() { return libbase::keypressed()>0 || libbase::interrupted(); };
};

const char *getlastargument(int *argc, char **argv[])
   {
   // read & swallow argument
   const char *a = (*argv)[*argc-1];
   (*argv)[*argc-1] = NULL;
   (*argc)--;
   return a;
   }

libcomm::experiment *createsystem(int *argc, char **argv[])
   {
   if(*argc < 2)
      {
      cerr << "Usage: " << (*argv)[0] << " [<other parameters>] <system>\n";
      exit(1);
      }
   // load system
   std::ifstream file(getlastargument(argc, argv));
   libcomm::experiment *system;
   file >> system;
   return system;
   }

double getminerror(int *argc, char **argv[])
   {
   if(*argc < 2)
      {
      cerr << "Usage: " << (*argv)[0] << " [<other parameters>] <min_error>\n";
      exit(1);
      }
   return atof(getlastargument(argc, argv));
   }

libbase::vector<double> getparameterset(int *argc, char **argv[])
   {
   if(*argc < 5)
      {
      cerr << "Usage: " << (*argv)[0] << " [<other parameters>] <-lin|-log> <min> <max> <step>\n";
      exit(1);
      }
   // read range specification
   const double Pstep = atof(getlastargument(argc, argv));
   const double Pmax = atof(getlastargument(argc, argv));
   const double Pmin = atof(getlastargument(argc, argv));
   const char *type = getlastargument(argc, argv);
   // validate range
   double steps = 0;
   bool linear = true;
   if(strcmp(type,"-lin")==0)
      {
      steps = floor((Pmax-Pmin)/Pstep)+1;
      linear = true;
      }
   else if(strcmp(type,"-log")==0)
      {
      if(Pmax==0 && Pmin==0)
         steps = 1;
      else
         steps = floor((log(Pmax)-log(Pmin))/log(Pstep))+1;
      linear = false;
      }
   else
      {
      cerr << "Invalid range type: " << type << "\n";
      exit(1);
      }
   if(!(steps >= 1 && steps <= 65535))
      {
      cerr << "Range does not converge: " << Pmin << ':' << Pstep << ':' << Pmax << '\n';
      exit(1);
      }
   // create required range
   const int count = int(steps);
   libbase::vector<double> Pset(count);
   Pset(0) = Pmin;
   for(int i=1; i<count; i++)
      Pset(i) = linear ? (Pset(i-1) + Pstep) : (Pset(i-1) * Pstep);
   return Pset;
   }

double getconfidence(int *argc, char **argv[])
   {
   if(*argc < 2)
      return 0.90;
   return atof(getlastargument(argc, argv));
   }

double gettolerance(int *argc, char **argv[])
   {
   if(*argc < 2)
      return 0.15;
   return atof(getlastargument(argc, argv));
   }

int main(int argc, char *argv[])
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;

   libbase::timer tmain("Main timer");

   // Create estimator object and initilize cluster, default priority
   mymontecarlo estimator;
   estimator.enable(&argc, &argv);

   // Simulation system & parameters, in reverse order
   libcomm::experiment *system = createsystem(&argc, &argv);
   estimator.initialise(system);
   const double min_error = getminerror(&argc, &argv);
   libbase::vector<double> Pset = getparameterset(&argc, &argv);
   const double confidence = getconfidence(&argc, &argv);
   const double accuracy = gettolerance(&argc, &argv);
   estimator.set_confidence(confidence);
   estimator.set_accuracy(accuracy);

   // Print information on the statistical accuracy of results being worked
   cout << "#% " << system->description() << "\n";
   cout << "#% Tolerance: " << 100*accuracy << "%\n";
   cout << "#% Confidence: " << 100*confidence << "%\n";
   cout << "#% Date: " << libbase::timer::date() << "\n";
   cout << "#% URL: " << __WCURL__ << "\n";
   cout << "#% Version: " << __WCVER__ << "\n";
   cout << "#\n";
   // Print results header
   cout << "# Par";
   for(int i=0; i<system->count(); i++)
      cout << "\t" << system->result_description(i) << "\tTol";
   cout << "\tSamples\tCPUtime\n" << flush;

   // Work out the following for every SNR value required
   for(int i=0; i<Pset.size(); i++)
      {
      system->set_parameter(Pset(i));

      cerr << "Simulating system at parameter = " << Pset(i) << "\n";
      libbase::vector<double> estimate, tolerance;
      estimator.estimate(estimate, tolerance);

      cerr << "Statistics: " << setprecision(4)
         << estimator.get_samplecount() << " frames in " << estimator.get_timer() << " - "
         << estimator.get_samplecount()/estimator.get_timer().elapsed() << " frames/sec\n";

      if(estimator.get_samplecount() > 0)
         {
         cout << Pset(i);
         for(int i=0; i<system->count(); i++)
            cout << "\t" << estimate(i) << "\t" << estimate(i)*tolerance(i);
         cout << "\t" << estimator.get_samplecount();
         cout << "\t" << estimator.getcputime() << "\n" << flush;
         }

      // handle pre-mature breaks
      if(estimator.interrupt() || estimate.min()<min_error)
         break;
      }

   return 0;
   }
