#include "turbo.h"

#include "logrealfast.h"
#include "rscc.h"

#include "helical.h"
#include "vector.h"
#include "puncture_stipple.h"
#include "mpsk.h"
#include "awgn.h"
#include "randgen.h"
#include "commsys.h"
#include "montecarlo.h"
#include "timer.h"
#include "masterslave.h"

#include <math.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <iomanip>

/*!
   \file
   \brief   SPECturbo benchmark
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 2.44 (7 Apr 2002)
   in order to allow profiling with more accuracy, modified the main simulation loop
   to stop after two conditions are satisfied: the first is the existing time condition
   (wall-clock), and the second is the minimum number of frames to simulate. This ensures
   that the profiling information is more valid since a better proportion of executable
   time will be in the simulation loop.

   \version 2.45 (12 Jun 2002)
   added a second command-line parameter "-q" which forces the program to output nothing
   to stdout except for the final speed metric.

   \version 2.46 (25 Jul 2006)
   after fixing a bug introduced recently in the turbo 'encode' process, the official
   list of standard results has now been updated so that a 0% deviation is expected
   (rather than the -5% on BER that was expected in the earlier version).

   \version 2.50 (27 Jul 2006)
   first version based on montecarlo class, rather than direct sampling of the commsys
   class. The most notable improvement is that this automatically incorporates the use
   of MPI for benchmarking clusters / multi-processor systems. Notably here, the final
   benchmark of frames/CPUsec has now become frames/sec, as scaling by usage is no
   longer meaningful.

   \version 2.51 (1 Aug 2006)
   following the update to bcjr, where the alpha and beta metrics are normalized, this
   version of SPECturbo now uses the double-precision based turbo and bcjr algorithms,
   resulting in more than 6x increase in speed. Consequently, the simulation tolerance
   limits had to be tightened.

   \version 2.52 (30 Oct 2006)
   - updated to use library namespaces.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 2.53 (10 Nov 2006)
   - removed use of "using namespace" for libbase and libcomm.

   \version 2.60 (20 Apr 2007)
   - converted to use masterslave instead of cmpi
   - TODO: fix argument handling (masterslave should do its thing before the main function)
   - TODO: fix system so that it can work in single-task mode ?

   \version 2.61 (24-25 Apr 2007)
   - removed call to masterslave::disable(), to conform with masterslave 1.10
   - updated to conform with montecarlo 1.31
   - refactored codec creation to occur within a separate function and to create
    all system components on the heap.

   \version 2.62 (24-25 Apr 2007)
   - added VERSION constant
   - modified turbo codec creation parameters (removed simile flag)
   - added consideration of SER
*/

const char *VERSION = "2.62";

using std::cout;
using std::setprecision;

class mymontecarlo : public libcomm::montecarlo {
protected:
   bool interrupt() { return get_timer().elapsed() > timeout && get_samplecount() > 50; };
public:
   double   timeout;
};

libcomm::commsys createsystem()
   {
   // Encoder (from generator matrix)
   const int k=1, n=2, m=2;
   libbase::matrix<libbase::bitfield> gen(k, n);
   gen(0, 0) = "111";
   gen(0, 1) = "101";
   libcomm::rscc encoder(gen);
   // Block interleaver parameters
   const int rows = 13, cols = 12;
   const int tau = rows*cols + m;
   // Helical interleaver (from matrix size, hence block size)
   libbase::vector<libcomm::interleaver *> inter(1);
   inter(0) = new libcomm::helical(tau, rows, cols);
   // Channel Codec (punctured, iterations, endatzero)
   libcomm::turbo<double> *codec = new libcomm::turbo<double>(encoder, tau, inter, 10, true);
   // Stipple puncturing
   const int sets = inter.size()+1;
   libcomm::puncture_stipple *punc = new libcomm::puncture_stipple(tau, k+sets*(n-k));
   // Modulation scheme
   libcomm::mpsk *modem = new libcomm::mpsk(2);
   // Channel Model
   libcomm::awgn *chan = new libcomm::awgn;
   // Source Generator
   libbase::randgen *src = new libbase::randgen;
   // The complete communication system
   return libcomm::commsys(src, codec, modem, punc, chan);
   }

int main(int argc, char *argv[])
   {
   // Create estimator object and initilize cluster, default priority
   mymontecarlo estimator;
   estimator.enable(&argc, &argv);

   // Simulation parameters
   const double SNR = 0.5;
   const double simtime = argc > 1 ? atoi(argv[1]) : 60;
   const bool quiet = argc > 2 ? (strcmp(argv[2],"-q")==0) : false;
   const double confidence = 0.999;
   const double accuracy = 0.001;
   // Set up the estimator
   libcomm::commsys system = createsystem();
   estimator.initialise(&system);
   estimator.set_confidence(confidence);
   estimator.set_accuracy(accuracy);

   // Write some information on the code
   if(!quiet)
      {
      cout << "\n";
      cout << "System Used:\n";
      cout << "~~~~~~~~~~~~\n";
      cout << system.description() << "\n";
      cout << "Tolerance: " << 100*accuracy << "%\n";
      cout << "Confidence: " << 100*confidence << "%\n";
      cout << "Date: " << libbase::timer::date() << "\n";
      cout << "Simulating system at Eb/No = " << SNR << "\n";
      cout << "\n";
      }

   // Work out at the SNR value required
   system.set(SNR);
   // Time the simulation
   estimator.timeout = simtime;
   libbase::vector<double> estimate, tolerance;
   estimator.estimate(estimate, tolerance);

   // Tabulate standard results
   const double std[] = {0.0924156, 0.0924156, 0.993763, \
                         0.073373, 0.073373, 0.894948, \
                         0.0671458, 0.0671458, 0.798102, \
                         0.0646009, 0.0646009, 0.740787, \
                         0.0634388, 0.0634388, 0.70745, \
                         0.0628046, 0.0628046, 0.690988, \
                         0.0622686, 0.0622686, 0.679999, \
                         0.0620079, 0.0620079, 0.670621, \
                         0.0619153, 0.0619153, 0.666856, \
                         0.0618174, 0.0618174, 0.662668};

   // Print results (for confirming accuracy)
   if(!quiet)
      {
      cout << "\n";
      cout << "Results: (BER, SER, FER)\n";
      cout << "~~~~~~~~~~~~~~~~~~~~~~~~\n";
      for(int j=0; j<system.count(); j+=3)
         {
         cout << setprecision(6) << estimate(j) << " (";
         cout << setprecision(3) << 100*(estimate(j)-std[j])/std[j] << "%)\t";
         cout << setprecision(6) << estimate(j+1) << " (";
         cout << setprecision(3) << 100*(estimate(j+1)-std[j+1])/std[j+1] << "%)\t";
         cout << setprecision(6) << estimate(j+2) << " (";
         cout << setprecision(3) << 100*(estimate(j+2)-std[j+2])/std[j+2] << "%)\n";
         }
      cout << "\n";
      }

   // Output timing statistics
   const int frames = estimator.get_samplecount();
   if(!quiet)
      {
      cout << "SPECturbo Version " << VERSION << "\n";
      cout << "Statistics: " << frames << " frames in " << estimator.get_timer() << ".\n";
      }
   cout << "SPECturbo: " << setprecision(4) << frames/estimator.get_timer().elapsed() << " frames/sec\n";
   return 0;
   }
