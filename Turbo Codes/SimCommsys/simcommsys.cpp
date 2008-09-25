#include "randgen.h"
#include "serializer_libcomm.h"
#include "commsys_simulator.h"
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

std::string getresultsfile(int *argc, char **argv[])
   {
   if(*argc < 2)
      {
      cerr << "Usage: " << (*argv)[0] << " [<other parameters>] <resultsfile>\n";
      exit(1);
      }
   return getlastargument(argc, argv);
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
   estimator.set_resultsfile(getresultsfile(&argc, &argv));
   libcomm::experiment *system = createsystem(&argc, &argv);
   estimator.bind(system);
   const double min_error = getminerror(&argc, &argv);
   libbase::vector<double> Pset = getparameterset(&argc, &argv);
   estimator.set_confidence(getconfidence(&argc, &argv));
   estimator.set_accuracy(gettolerance(&argc, &argv));

   // Work out the following for every SNR value required
   for(int i=0; i<Pset.size(); i++)
      {
      system->set_parameter(Pset(i));

      cerr << "Simulating system at parameter = " << Pset(i) << "\n";
      libbase::vector<double> result, tolerance;
      estimator.estimate(result, tolerance);

      cerr << "Statistics: " << setprecision(4)
         << estimator.get_samplecount() << " frames in " << estimator.get_timer() << " - "
         << estimator.get_samplecount()/estimator.get_timer().elapsed() << " frames/sec\n";

      // handle pre-mature breaks
      if(estimator.interrupt() || result.min()<min_error)
         break;
      }

   return 0;
   }
