#include "config.h"
#include "randgen.h"
#include "truerand.h"
#include "serializer_libcomm.h"
#include "commsys.h"
#include "timer.h"

#include <iostream>

/*!
   \file
   \brief   Error Event Analysis for Experiments
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

using std::cout;
using std::cerr;

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

double getparameter(int *argc, char **argv[])
   {
   if(*argc < 2)
      {
      cerr << "Usage: " << (*argv)[0] << " [<other parameters>] <parameter>\n";
      exit(1);
      }
   return atof(getlastargument(argc, argv));
   }

void seed_experiment(libcomm::experiment *system)
   {
   libbase::truerand trng;
   libbase::randgen prng;
   const libbase::int32u seed = trng.ival();
   prng.seed(seed);
   system->seedfrom(prng);
   cerr << "Seed: " << seed << "\n";
   }

int main(int argc, char *argv[])
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;

   libbase::timer tmain("Main timer");

   // Simulation system & parameters, in reverse order
   libcomm::experiment *system = createsystem(&argc, &argv);
   system->set_parameter(getparameter(&argc, &argv));

   // Initialise running values
   system->reset();
   seed_experiment(system);
   cerr << "Simulating system at parameter = " << system->get_parameter() << "\n";
   // Simulate, waiting for an error event
   libbase::vector<double> result;
   do {
      system->sample(result);
      system->accumulate(result);
      } while(result.min() == 0);

   return 0;
   }
