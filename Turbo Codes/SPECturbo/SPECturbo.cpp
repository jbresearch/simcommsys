/*!
   \file
   \brief   SPECturbo benchmark
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \note Following the update to bcjr, where the alpha and beta metrics are
         normalized, SPECturbo now uses the double-precision based turbo and
         bcjr algorithms, resulting in more than 6x increase in speed.
*/

#include "serializer_libcomm.h"
#include "montecarlo.h"
#include "timer.h"

#include <boost/program_options.hpp>

#include <math.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace po = boost::program_options;

class mymontecarlo : public libcomm::montecarlo {
protected:
   bool interrupt() { return get_timer().elapsed() > timeout && get_samplecount() > 50; };
public:
   double timeout;
};

libcomm::experiment *createsystem()
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;

   const std::string systemstring = 
      "commsys_simulator<sigspace>\n"
      "commsys<sigspace>\n"
      "awgn\n"
      "mpsk\n"
      "2\n"
      "map_stipple\n"
      "158\n"
      "2\n"
      "turbo<double>\n"
      "1\n"
      "rscc\n"
      "1\t2\n"
      "111\n"
      "101\n"
      "158\n"
      "2\n"
      "flat\n"
      "158\n"
      "helical\n"
      "158\n"
      "13\n"
      "12\n"
      "1\n"
      "0\n"
      "0\n"
      "10\n";

   // load system from string representation
   libcomm::experiment *system;
   std::istringstream is(systemstring);
   is >> system;
   // check for errors in loading system
   libbase::verifycompleteload(is);
   return system;
   }

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::setprecision;

   // Create estimator object and initilize cluster, default priority
   mymontecarlo estimator;
   estimator.enable(&argc, &argv);

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()
      ("help", "print this help message")
      ("quiet", po::bool_switch(),
         "suppress all output except benchmark")
      ("snr", po::value<double>()->default_value(0.5),
         "signal to noise ratio")
      ("time", po::value<double>()->default_value(60),
         "benchmark duration in seconds")
      //("system-file,i", po::value<std::string>(),
      //   "file containing system description")
      ("confidence", po::value<double>()->default_value(0.999),
         "confidence level (e.g. 0.90 for 90%)")
      ("tolerance", po::value<double>()->default_value(0.001),
         "confidence interval (e.g. 0.15 for +/- 15%)")
      ;
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if(vm.count("help"))
      {
      cout << desc << "\n";
      return 0;
      }

   // Simulation parameters
   const double SNR = vm["snr"].as<double>();
   const double simtime = vm["time"].as<double>();
   const bool quiet = vm["quiet"].as<bool>();
   const double confidence = vm["confidence"].as<double>();
   const double accuracy = vm["tolerance"].as<double>();
   // Set up the estimator
   libcomm::experiment *system = createsystem();
   estimator.bind(system);
   estimator.set_confidence(confidence);
   estimator.set_accuracy(accuracy);

   // Write some information on the code
   if(!quiet)
      {
      cout << "\n";
      cout << "System Used:\n";
      cout << "~~~~~~~~~~~~\n";
      cout << system->description() << "\n";
      //cout << "Rate: " << system-> << "\n";
      cout << "Tolerance: " << 100*accuracy << "%\n";
      cout << "Confidence: " << 100*confidence << "%\n";
      cout << "Date: " << libbase::timer::date() << "\n";
      cout << "Simulating system at Eb/No = " << SNR << "\n";
      cout << "\n";
      }

   // Work out at the SNR value required
   system->set_parameter(SNR);
   // Time the simulation
   estimator.timeout = simtime;
   libbase::vector<double> estimate, tolerance;
   estimator.estimate(estimate, tolerance);

   // Tabulate standard results
   const double std[] = {0.0924156, 0.993763, \
                         0.073373, 0.894948, \
                         0.0671458, 0.798102, \
                         0.0646009, 0.740787, \
                         0.0634388, 0.70745, \
                         0.0628046, 0.690988, \
                         0.0622686, 0.679999, \
                         0.0620079, 0.670621, \
                         0.0619153, 0.666856, \
                         0.0618174, 0.662668};

   // Print results (for confirming accuracy)
   if(!quiet)
      {
      cout << "\n";
      cout << "Results: (SER, FER)\n";
      cout << "~~~~~~~~~~~~~~~~~~~\n";
      for(int j=0; j<system->count(); j+=2)
         {
         cout << setprecision(6) << estimate(j) << " (";
         cout << setprecision(3) << 100*(estimate(j)-std[j])/std[j] << "%)\t";
         cout << setprecision(6) << estimate(j+1) << " (";
         cout << setprecision(3) << 100*(estimate(j+1)-std[j+1])/std[j+1] << "%)\n";
         }
      cout << "\n";
      }

   // Output timing statistics
   const libbase::int64u frames = estimator.get_samplecount();
   if(!quiet)
      {
      cout << "URL: " << __WCURL__ << "\n";
      cout << "Version: " << __WCVER__ << "\n";
      cout << "Statistics: " << frames << " frames in " << estimator.get_timer() << ".\n";
      }
   cout << "SPECturbo: " << setprecision(4) << frames/estimator.get_timer().elapsed() << " frames/sec\n";
   return 0;
   }
