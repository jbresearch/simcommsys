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

// Standard system and respective results

const std::string std_systemstring = 
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
   "flat<double>\n"
   "158\n"
   "helical<double>\n"
   "158\n"
   "13\n"
   "12\n"
   "1\n"
   "0\n"
   "0\n"
   "10\n";

const double std_result[] = {
   0.0924156, 0.993763, \
   0.073373, 0.894948, \
   0.0671458, 0.798102, \
   0.0646009, 0.740787, \
   0.0634388, 0.70745, \
   0.0628046, 0.690988, \
   0.0622686, 0.679999, \
   0.0620079, 0.670621, \
   0.0619153, 0.666856, \
   0.0618174, 0.662668 };

class mymontecarlo : public libcomm::montecarlo {
protected:
   bool interrupt() { return get_timer().elapsed() > timeout; };
public:
   double timeout;
};

libcomm::experiment *loadandverify(std::istream& file)
   {
   libcomm::experiment *system;
   file >> system;
   libbase::verifycompleteload(file);
   return system;
   }

libcomm::experiment *createsystem()
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;
   // load system from string representation
   std::istringstream is(std_systemstring);
   return loadandverify(is);
   }

libcomm::experiment *createsystem(const std::string& fname)
   {
   // load system from string representation
   std::ifstream file(fname.c_str());
   return loadandverify(file);
   }

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::setprecision;

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()
      ("help", "print this help message")
      ("quiet,q", po::bool_switch(),
         "suppress all output except benchmark")
      ("priority,p", po::value<int>()->default_value(10),
         "process priority")
      ("endpoint,e", po::value<std::string>()->default_value("local"),
         "- 'local', for local-computation model\n"
         "- ':port', for server-mode, bound to given port\n"
         "- 'hostname:port', for client-mode connection")
      ("time,t", po::value<double>()->default_value(60),
         "benchmark duration in seconds")
      ("snr", po::value<double>()->default_value(0.5),
         "signal to noise ratio")
      ("system-file,i", po::value<std::string>(),
         "file containing system description")
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
      return 1;
      }

   // Create estimator object and initilize cluster
   mymontecarlo estimator;
   estimator.enable(vm["endpoint"].as<std::string>(), vm["quiet"].as<bool>(), vm["priority"].as<int>());
   // Set up the estimator
   libcomm::experiment *system;
   if(vm.count("system-file"))
      system = createsystem(vm["system-file"].as<std::string>());
   else
      system = createsystem();
   estimator.bind(system);
   estimator.set_confidence(vm["confidence"].as<double>());
   estimator.set_accuracy(vm["tolerance"].as<double>());
   estimator.timeout = vm["time"].as<double>();
   // Work out at the SNR value required
   system->set_parameter(vm["snr"].as<double>());

   // Perform the simulation
   libbase::vector<double> estimate, tolerance;
   estimator.estimate(estimate, tolerance);
   const libbase::int64u frames = estimator.get_samplecount();

   if(!vm["quiet"].as<bool>())
      {
      // Write some information on the code
      cout << "\n\n";
      cout << "System Used:\n";
      cout << "~~~~~~~~~~~~\n";
      cout << system->description() << "\n";
      //cout << "Rate: " << system-> << "\n";
      cout << "Tolerance: " << 100*estimator.get_accuracy() << "%\n";
      cout << "Confidence: " << 100*estimator.get_confidence() << "%\n";
      cout << "Date: " << libbase::timer::date() << "\n";
      cout << "Simulating system at Eb/No = " << system->get_parameter() << "\n";

      // Print results (for confirming accuracy)
      cout << "\n";
      cout << "Results: (SER, FER)\n";
      cout << "~~~~~~~~~~~~~~~~~~~\n";
      for(int j=0; j<system->count(); j+=2)
         {
         cout << setprecision(6) << estimate(j);
         if(!vm.count("system-file"))
            cout << " (" << setprecision(3) << 100*(estimate(j)-std_result[j])/std_result[j] << "%)";
         cout << "\t";
         cout << setprecision(6) << estimate(j+1);
         if(!vm.count("system-file"))
            cout << " (" << setprecision(3) << 100*(estimate(j+1)-std_result[j+1])/std_result[j+1] << "%)";
         cout << "\n";
         }

      // Output timing statistics
      cout << "\n";
      cout << "URL: " << __WCURL__ << "\n";
      cout << "Version: " << __WCVER__ << "\n";
      cout << "Statistics: " << frames << " frames in " << estimator.get_timer() << ".\n";
      }

   // Output overall benchmark
   cout << "SPECturbo: " << setprecision(4) << frames/estimator.get_timer().elapsed() << " frames/sec\n";
   return 0;
   }
