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
*/

using std::cout;
using std::cerr;
using std::setprecision;
using std::flush;

const libcomm::serializer_libcomm g_serializer_libcomm;

class mymontecarlo : public libcomm::montecarlo {
public:
   // make interrupt function public to allow use by main program
   bool interrupt() { return libbase::keypressed()>0 || libbase::interrupted(); };
};

libcomm::commsys *createsystem(const char *filename)
   {
   std::ifstream file(filename);
   // Channel Model
   libcomm::channel *chan;
   file >> chan;
   // Modulation scheme
   libcomm::modulator *modem;
   file >> modem;
   // Channel Codec
   libcomm::codec *codec;
   file >> codec;
   // Source Generator
   libbase::randgen *src = new libbase::randgen;
   // The complete communication system
   return new libcomm::commsys(src, codec, modem, NULL, chan);
   }

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");

   // Create estimator object and initilize cluster, default priority
   mymontecarlo estimator;
   estimator.enable(&argc, &argv);

   // Simulation parameters
   if(argc < 6)
      {
      cerr << "Usage: " << argv[0] << " SNRmin SNRmax SNRstep ERmin System\n";
      exit(1);
      }
   const double SNRmin = atof(argv[1]);
   const double SNRmax = atof(argv[2]);
   const double SNRstep = atof(argv[3]);
   if(SNRmax < SNRmin || SNRstep <= 0)
      {
      cerr << "Invalid SNR parameters: " << SNRmin << ", " << SNRmax << ", " << SNRstep << "\n";
      exit(1);
      }
   const double ERmin = atof(argv[4]);
   const double confidence = 0.90;
   const double accuracy = 0.15;
   // Set up the estimator
   libcomm::commsys *system = createsystem(argv[5]);
   estimator.initialise(system);
   estimator.set_confidence(confidence);
   estimator.set_accuracy(accuracy);

   // Print information on the statistical accuracy of results being worked
   cout << "#% " << system->description() << "\n";
   cout << "#% Code Rate: " << system->getcodec()->rate() << "\n";
   cout << "#% Modulation Rate: " << system->getmodem()->rate() << "\n";
   cout << "#% Tolerance: " << 100*accuracy << "%\n";
   cout << "#% Confidence: " << 100*confidence << "%\n";
   cout << "#% Date: " << libbase::timer::date() << "\n";
   cout << "#\n" << flush;

   // Work out the following for every SNR value required
   for(double SNR = SNRmin; SNR <= SNRmax; SNR += SNRstep)
      {
      system->set_parameter(SNR);

      cerr << "Simulating system at Eb/No = " << SNR << "\n";
      libbase::vector<double> estimate, tolerance;
      estimator.estimate(estimate, tolerance);

      cerr << "Statistics: " << setprecision(4) \
         << estimator.get_samplecount() << " frames in " << estimator.get_timer() << " - " \
         << estimator.get_samplecount()/estimator.get_timer().elapsed() << " frames/sec\n";

      cout << SNR;
      for(int i=0; i<system->count(); i++)
         cout << "\t" << estimate(i) << "\t" << estimate(i)*tolerance(i);
      cout << "\t" << estimator.get_samplecount() << "\n" << flush;

      // handle pre-mature breaks
      if(estimator.interrupt() || estimate.min()<ERmin)
         break;
      }

   return 0;
   }
