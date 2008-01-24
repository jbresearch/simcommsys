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

   \version 1.10 (30 Oct 2006)
   - updated to use library namespaces.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.11 (10 Nov 2006)
   - removed use of "using namespace" for libbase and libcomm.

   \version 1.20 (20 Apr 2007)
   - updated to use masterslave instead of cmpi class, to support socket-based model

   \version 1.21 (24-25 Apr 2007)
   - removed call to masterslave::disable(), to conform with masterslave 1.10
   - updated to conform with montecarlo 1.31
   - refactored codec creation to occur within a separate function and to create
    all system components on the heap.

   \version 1.22 (2 Nov 2007)
   - modified input scheme so that the file also contains the channel and modem
    before the codec; as a result, previous codec files cannot be used directly.

   \version 1.23 (16 Nov 2007)
   - added output flushing to guarantee results are always on file.

   \version 1.24 (29-30 Nov 2007)
   - added printing of Code and Modulation rates.

   \version 1.25 (21 Dec 2007)
   - added minimum error rate cutoff, including command-line parameter

   \version 1.26 (22 Jan 2008)
   - modified createsystem() to return a pointer to a commsys object, created
     on the heap, rather than a copy created on the stack.
   - modified createsystem() to use commsys serialization instead of manually
     creating the components.
   - moved serializer_libcomm object from a global variable to a local one in
     main; this resolves the possibility where the serializer_libcomm is
     initialized before the trace object.

   \version 2.00 (24 Jan 2008)
   - Modified createsystem() to return an experiment instead of a commsys,
     generalizing this program to all simulation types; consequently, this
     requires the addition of a line to system files specifying the system
     type.
   - Refactored, renaming variabled to indicate that the simulation parameter
     is no longer tied to SNR values.
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

libcomm::experiment *createsystem(const char *filename)
   {
   std::ifstream file(filename);
   libcomm::experiment *system;
   file >> system;
   return system;
   }

int main(int argc, char *argv[])
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;

   libbase::timer tmain("Main timer");

   // Create estimator object and initilize cluster, default priority
   mymontecarlo estimator;
   estimator.enable(&argc, &argv);

   // Simulation parameters
   if(argc < 7)
      {
      cerr << "Usage: " << argv[0] << " <min> <max> <step> <min_error> <system>\n";
      exit(1);
      }
   const double Pmin = atof(argv[1]);
   const double Pmax = atof(argv[2]);
   const double Pstep = atof(argv[3]);
   if(Pmax < Pmin || Pstep <= 0)
      {
      cerr << "Invalid Parameters: " << Pmin << ", " << Pmax << ", " << Pstep << "\n";
      exit(1);
      }
   const double min_error = atof(argv[4]);
   const double confidence = 0.90;
   const double accuracy = 0.15;
   // Set up the estimator
   libcomm::experiment *system = createsystem(argv[5]);
   estimator.initialise(system);
   estimator.set_confidence(confidence);
   estimator.set_accuracy(accuracy);

   // Print information on the statistical accuracy of results being worked
   cout << "#% " << system->description() << "\n";
   cout << "#% Tolerance: " << 100*accuracy << "%\n";
   cout << "#% Confidence: " << 100*confidence << "%\n";
   cout << "#% Date: " << libbase::timer::date() << "\n";
   cout << "#\n" << flush;

   // Work out the following for every SNR value required
   for(double P = Pmin; P <= Pmax; P += Pstep)
      {
      system->set_parameter(P);

      cerr << "Simulating system at parameter = " << P << "\n";
      libbase::vector<double> estimate, tolerance;
      estimator.estimate(estimate, tolerance);

      cerr << "Statistics: " << setprecision(4) \
         << estimator.get_samplecount() << " frames in " << estimator.get_timer() << " - " \
         << estimator.get_samplecount()/estimator.get_timer().elapsed() << " frames/sec\n";

      cout << P;
      for(int i=0; i<system->count(); i++)
         cout << "\t" << estimate(i) << "\t" << estimate(i)*tolerance(i);
      cout << "\t" << estimator.get_samplecount() << "\n" << flush;

      // handle pre-mature breaks
      if(estimator.interrupt() || estimate.min()<min_error)
         break;
      }

   return 0;
   }
