/*!
 * \file
 * 
 * Copyright (c) 2010 Johann A. Briffa
 * 
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "serializer_libcomm.h"
#include "montecarlo.h"
#include "timer.h"

#include <boost/program_options.hpp>

#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace SPECturbo {

namespace po = boost::program_options;

//! Standard benchmark system

const std::string std_systemstring = "# simulator\n"
   "commsys_simulator<sigspace>\n"
   "## communication system\n"
   "commsys<sigspace>\n"
   "### channel\n"
   "awgn\n"
   "### modem\n"
   "mpsk\n"
   "2   # alphabet size in symbols\n"
   "### symbol mapper\n"
   "map_stipple<vector>\n"
   "2   # stipple stride\n"
   "### codec\n"
   "turbo<double>\n"
   "2   # format version\n"
   "#### encoder (fsm)\n"
   "rscc\n"
   "1\t2\n"
   "111\n"
   "101\n"
   "2   # number of sets\n"
   "#### interleaver 0\n"
   "flat<double>\n"
   "158 # interleaver size\n"
   "#### interleaver 1\n"
   "helical<double>\n"
   "158 # interleaver size\n"
   "13  # rows\n"
   "12  # cols\n"
   "1   # terminated?\n"
   "0   # circular?\n"
   "0   # parallel decoder?\n"
   "10  # number of iterations\n";

//! Standard benchmark result set

const double std_result[] = {0.0924156, 0.993763, 0.073373, 0.894948,
      0.0671458, 0.798102, 0.0646009, 0.740787, 0.0634388, 0.70745, 0.0628046,
      0.690988, 0.0622686, 0.679999, 0.0620079, 0.670621, 0.0619153, 0.666856,
      0.0618174, 0.662668};

class mymontecarlo : public libcomm::montecarlo {
protected:
   bool interrupt()
      {
      return montecarlo::interrupt() || get_timer().elapsed() > timeout;
      }
public:
   double timeout;
};

/*!
 * \brief   SPECturbo benchmark
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * \note Following the update to bcjr, where the alpha and beta metrics are
 * normalized, SPECturbo now uses the double-precision based turbo and
 * bcjr algorithms, resulting in more than 6x increase in speed.
 */

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::setprecision;

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("quiet,q", po::bool_switch(),
         "suppress all output except benchmark");
   desc.add_options()("priority,p", po::value<int>()->default_value(10),
         "process priority");
   desc.add_options()("endpoint,e", po::value<std::string>()->default_value(
         "local"), "- 'local', for local-computation model\n"
      "- ':port', for server-mode, bound to given port\n"
      "- 'hostname:port', for client-mode connection");
   desc.add_options()("time,t", po::value<double>()->default_value(60),
         "benchmark duration in seconds");
   desc.add_options()("parameter,r", po::value<double>()->default_value(0.5),
         "channel parameter (e.g. SNR)");
   desc.add_options()("system-file,i", po::value<std::string>(),
         "file containing system description");
   desc.add_options()("confidence", po::value<double>()->default_value(0.999),
         "confidence level (e.g. 0.90 for 90%)");
   desc.add_options()("tolerance", po::value<double>()->default_value(0.001),
         "confidence interval (e.g. 0.15 for +/- 15%)");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      cout << desc << std::endl;
      return 1;
      }

   // Create estimator object and initilize cluster
   mymontecarlo estimator;
   estimator.enable(vm["endpoint"].as<std::string> (), vm["quiet"].as<bool> (),
         vm["priority"].as<int> ());
   // Set up the estimator
   libcomm::experiment *system;
   if (vm.count("system-file"))
      system = libcomm::loadfromfile<libcomm::experiment>(vm["system-file"].as<
            std::string> ());
   else
      system = libcomm::loadfromstring<libcomm::experiment>(std_systemstring);
   estimator.bind(system);
   estimator.set_confidence(vm["confidence"].as<double> ());
   estimator.set_accuracy(vm["tolerance"].as<double> ());
   estimator.timeout = vm["time"].as<double> ();
   // Work out at the SNR value required
   system->set_parameter(vm["parameter"].as<double> ());

   // Print some debug information
   libbase::trace << system->description() << std::endl;

   // Perform the simulation
   libbase::vector<double> estimate, tolerance;
   estimator.estimate(estimate, tolerance);
   const libbase::int64u frames = estimator.get_samplecount();

   if (!vm["quiet"].as<bool> ())
      {
      // Write some information on the code
      cout << std::endl << std::endl;
      cout << "System Used:" << std::endl;
      cout << "~~~~~~~~~~~~" << std::endl;
      cout << system->description() << std::endl;
      //cout << "Rate: " << system-> << std::endl;
      cout << "Tolerance: " << 100 * estimator.get_accuracy() << "%"
            << std::endl;
      cout << "Confidence: " << 100 * estimator.get_confidence() << "%"
            << std::endl;
      cout << "Date: " << libbase::timer::date() << std::endl;
      // TODO: add method to system to get parameter name
      cout << "Simulating at system parameter = " << system->get_parameter()
            << std::endl;

      // Print results (for confirming accuracy)
      cout << std::endl;
      cout << "Results:" << std::endl;
      cout << "~~~~~~~~" << std::endl;
      for (int j = 0; j < system->count(); j++)
         {
         cout << system->result_description(j) << '\t';
         cout << setprecision(6) << estimate(j);
         if (!vm.count("system-file"))
            cout << "\t(" << setprecision(3) << 100 * (estimate(j)
                  - std_result[j]) / std_result[j] << "%)";
         cout << std::endl;
         }

      // Output timing statistics
      cout << std::endl;
      cout << "URL: " << __WCURL__ << std::endl;
      cout << "Version: " << __WCVER__ << std::endl;
      cout << "Statistics: " << frames << " frames in "
            << estimator.get_timer() << "." << std::endl;
      }

   // Output overall benchmark
   cout << "SPECturbo: " << setprecision(4) << frames
         / estimator.get_timer().elapsed() << " frames/sec" << std::endl;
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return SPECturbo::main(argc, argv);
   }
