
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

/*
  Version 1.10 (30 Oct 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.11 (10 Nov 2006)
  * removed use of "using namespace" for libbase and libcomm.

  Version 1.20 (20 Apr 2007)
  * updated to use masterslave instead of cmpi class, to support socket-based model
  
  Version 1.21 (24-25 Apr 2007)
  * removed call to masterslave::disable(), to conform with masterslave 1.10
  * updated to conform with montecarlo 1.31
  * refactored codec creation to occur within a separate function and to create
    all system components on the heap.
  
  Version 1.22 (2 Nov 2007)
  * modified input scheme so that the file also contains the channel and modem
    before the codec; as a result, previous codec files cannot be used directly.
  
  Version 1.23 (16 Nov 2007)
  * added output flushing to guarantee
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

libcomm::commsys createsystem(const char *filename)
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
   return libcomm::commsys(src, codec, modem, NULL, chan);
   }

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");
   
   // Create estimator object and initilize cluster, default priority
   mymontecarlo estimator;
   estimator.enable(&argc, &argv);

   // Simulation parameters
   if(argc < 5)
      {
      cerr << "Usage: " << argv[0] << " SNRmin SNRmax SNRstep System\n";
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
   const double confidence = 0.90;
   const double accuracy = 0.15;
   // Set up the estimator
   libcomm::commsys system = createsystem(argv[4]);
   estimator.initialise(&system);
   estimator.set_confidence(confidence);
   estimator.set_accuracy(accuracy);
      
   // Print information on the statistical accuracy of results being worked
   cout << "#% " << system.description() << "\n"; 
   cout << "#% Tolerance: " << 100*accuracy << "%\n";
   cout << "#% Confidence: " << 100*confidence << "%\n";
   cout << "#% Date: " << libbase::timer::date() << "\n";
   cout << "#\n" << flush;

   // Work out the following for every SNR value required
   for(double SNR = SNRmin; SNR <= SNRmax && !estimator.interrupt(); SNR += SNRstep)
      {
      system.set(SNR);

      cerr << "Simulating system at Eb/No = " << SNR << "\n";
      libbase::vector<double> estimate, tolerance;
      estimator.estimate(estimate, tolerance);
      
      cerr << "Statistics: " << setprecision(4) \
         << estimator.get_samplecount() << " frames in " << estimator.get_timer() << " - " \
         << estimator.get_samplecount()/estimator.get_timer().elapsed() << " frames/sec\n";

      cout << SNR;
      for(int i=0; i<system.count(); i++)
         cout << "\t" << estimate(i) << "\t" << estimate(i)*tolerance(i);
      cout << "\t" << estimator.get_samplecount() << "\n" << flush;
      }

   return 0;
   }
