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
 */

#include "version.h"
#include "serializer_libcomm.h"
#include "montecarlo.h"
#include "timer.h"

#include <boost/program_options.hpp>

#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace QuickSimulation {

namespace po = boost::program_options;

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
 * \brief   Quick Simulation
 * \author  Johann Briffa
 *
 * This program implements a quick simulation for a given system; this is
 * useful to benchmark the speed of the decoder and to obtain a quick estimate
 * for the performance of a code under given conditions.
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
   desc.add_options()("parameter,r", po::value<double>(),
         "channel parameter (e.g. SNR)");
   desc.add_options()("system-file,i", po::value<std::string>(),
         "file containing system description");
   desc.add_options()("seed,s", po::value<libbase::int32u>(),
         "system initialization seed (random if not stated)");
   desc.add_options()("confidence", po::value<double>()->default_value(0.999),
         "confidence level for computing margin of error (e.g. 0.90 for 90%)");
   desc.add_options()("relative-error", po::value<double>()->default_value(0.001),
         "target error margin, as a fraction of result mean (e.g. 0.15 for ±15%)");
   desc.add_options()("absolute-error", po::value<double>(),
         "target error margin, as an absolute value (e.g. 0.1 for ±0.1); "
               "overrides relative-error if specified");
   desc.add_options()("accumulated-result", po::value<double>(),
         "target accumulated result (i.e. result mean x sample count); "
               "overrides absolute and relative error if specified");
   desc.add_options()("min-samples", po::value<int>(),
         "minimum number of samples");
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
   switch (estimator.enable(vm["endpoint"].as<std::string> (), vm["quiet"].as<
         bool> (), vm["priority"].as<int> ()))
      {
      case mymontecarlo::mode_slave:
         break;

      case mymontecarlo::mode_local:
      case mymontecarlo::mode_master:
         {
         // If this is a server instance, check the remaining parameters
         if (vm.count("system-file") == 0 || vm.count("parameter") == 0)
            {
            cout << desc << std::endl;
            return 0;
            }
         // Set up the estimator
         libcomm::experiment *system;
         system = libcomm::loadfromfile<libcomm::experiment>(
               vm["system-file"].as<std::string> ());
         estimator.bind(system);
         estimator.set_confidence(vm["confidence"].as<double> ());
         if (vm.count("accumulated-result"))
            estimator.set_accumulated_result(vm["accumulated-result"].as<double>());
         else if (vm.count("absolute-error"))
            estimator.set_absolute_error(vm["absolute-error"].as<double>());
         else
            estimator.set_relative_error(vm["relative-error"].as<double>());
         if (vm.count("min-samples"))
            estimator.set_min_samples(vm["min-samples"].as<int>());
         if (vm.count("seed"))
            estimator.set_seed(vm["seed"].as<libbase::int32u> ());
         estimator.timeout = vm["time"].as<double> ();
         // Work out at the SNR value required
         system->set_parameter(vm["parameter"].as<double> ());

         // Print some debug information
         libbase::trace << system->description() << std::endl;

         // Perform the simulation
         libbase::vector<double> estimate, errormargin;
         estimator.estimate(estimate, errormargin);
         const libbase::int64u samples = estimator.get_samplecount();

         if (!vm["quiet"].as<bool> ())
            {
            // Write some information on the code
            cout << std::endl << std::endl;
            cout << "System Used:" << std::endl;
            cout << "~~~~~~~~~~~~" << std::endl;
            cout << system->description() << std::endl;
            //cout << "Rate: " << system-> << std::endl;
            cout << "Confidence Level: " << estimator.get_confidence_level()
                  << std::endl;
            cout << "Convergence Mode: " << estimator.get_convergence_mode()
                  << std::endl;
            cout << "Date: " << libbase::timer::date() << std::endl;
            // TODO: add method to system to get parameter name
            cout << "Simulating at system parameter = "
                  << system->get_parameter() << std::endl;

            // Print results (for confirming accuracy)
            cout << std::endl;
            cout << "Results:" << std::endl;
            cout << "~~~~~~~~" << std::endl;
            for (int j = 0; j < system->count(); j++)
               {
               cout << system->result_description(j) << '\t';
               cout << setprecision(6) << estimate(j);
               cout << "\t[±" << setprecision(3)
                     << fabs(100 * errormargin(j) / estimate(j)) << "%]";
               cout << std::endl;
               }

            // Output timing statistics
            cout << std::endl;
            cout << "Build: " << SIMCOMMSYS_BUILD << std::endl;
            cout << "Version: " << SIMCOMMSYS_VERSION << std::endl;
            cout << "Statistics: " << samples << " samples in "
                  << estimator.get_timer() << "." << std::endl;
            }

         // Output overall benchmark
         cout << "Simulation Speed: " << setprecision(4) << samples
               / estimator.get_timer().elapsed() << " samples/sec" << std::endl;

         // Destroy what was created on the heap
         delete system;
         }
         break;
      }
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return QuickSimulation::main(argc, argv);
   }
