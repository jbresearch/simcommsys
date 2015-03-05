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

#include "randgen.h"
#include "serializer_libcomm.h"
#include "experiment/binomial/commsys_simulator.h"
#include "montecarlo.h"
#include "masterslave.h"
#include "cputimer.h"

#include <boost/program_options.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>

namespace simcommsys {

using std::cout;
using std::cerr;
using std::setprecision;
namespace po = boost::program_options;

class mymontecarlo : public libcomm::montecarlo {
private:
   bool quiet; //!< Flag to disable intermediate displays
   bool hard_int; //!< Flag indicating a hard interrupt (stop completely)
   bool soft_int; //!< Flag indicating a soft interrupt (skip to next point)
public:
   mymontecarlo(bool quiet) :
         quiet(quiet), hard_int(false), soft_int(false)
      {
      }
   /*! \brief Conditional progress display
    *
    * If the object was set up to be quiet, then no display occurs, otherwise
    * use the default.
    */
   void display(const libbase::vector<double>& result,
         const libbase::vector<double>& errormargin) const
      {
      if (!quiet)
         libcomm::montecarlo::display(result, errormargin);
      }
   /*! \brief User-interrupt check (public to allow use by main program)
    * This function returns true if the user has requested a soft or hard
    * interrupt. As required by the interface, once it returns true, all
    * subsequent evaluations keep returning true again.
    * A soft interrupt can be checked for and reset; hard interrupts cannot.
    * Checks for user pressing 's' (soft), 'q' or Ctrl-C (hard).
    */
   bool interrupt()
      {
      if (hard_int || soft_int)
         return true;
      if (libbase::interrupted())
         hard_int = true;
      else if (libbase::keypressed() > 0)
         {
         const char k = libbase::readkey();
         hard_int = (k == 'q');
         soft_int = (k == 's');
         }
      return hard_int || soft_int;
      }
   /*! \brief Soft-interrupt check and reset
    * This should be called after interrupt(), and returns true if the
    * interrupt found was a soft interrupt. This method also resets the
    * soft interrupt condition when found.
    */
   bool interrupt_was_soft()
      {
      const bool result = soft_int;
      soft_int = false;
      return result;
      }
};

libcomm::experiment *createsystem(const std::string& fname)
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;
   // load system from string representation
   libcomm::experiment *system;
   std::ifstream file(fname.c_str(), std::ios_base::in | std::ios_base::binary);
   file >> system >> libbase::verifycomplete;
   return system;
   }

libbase::vector<double> getlinrange(double beg, double end, double step)
   {
   // validate range
   int steps = int(floor((end - beg) / step) + 1);
   assertalways(steps >= 1 && steps <= 65535);
   // create required range
   libbase::vector<double> pset(steps);
   pset(0) = beg;
   for (int i = 1; i < steps; i++)
      pset(i) = pset(i - 1) + step;
   return pset;
   }

libbase::vector<double> getlogrange(double beg, double end, double mul)
   {
   // validate range
   int steps = 0;
   if (end == 0 && beg == 0)
      steps = 1;
   else
      steps = int(floor((log(end) - log(beg)) / log(mul)) + 1);
   assertalways(steps >= 1 && steps <= 65535);
   // create required range
   libbase::vector<double> pset(steps);
   pset(0) = beg;
   for (int i = 1; i < steps; i++)
      pset(i) = pset(i - 1) * mul;
   return pset;
   }

/*!
 * \brief   Simulation of Communication Systems
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("quiet,q", po::bool_switch(),
         "suppress all output except benchmark");
   desc.add_options()("priority,p", po::value<int>()->default_value(10),
         "process priority");
   desc.add_options()("endpoint,e",
         po::value<std::string>()->default_value("local"),
         "- 'local', for local-computation model\n"
               "- ':port', for server-mode, bound to given port\n"
               "- 'hostname:port', for client-mode connection");
   desc.add_options()("system-file,i", po::value<std::string>(),
         "input file containing system description");
   desc.add_options()("results-file,o", po::value<std::string>(),
         "output file to hold results");
   desc.add_options()("start", po::value<double>(), "first parameter value");
   desc.add_options()("stop", po::value<double>(), "last parameter value");
   desc.add_options()("step", po::value<double>(),
         "parameter increment (for a linear range)");
   desc.add_options()("mul", po::value<double>(),
         "parameter multiplier (for a logarithmic range)");
   desc.add_options()("floor-min", po::value<double>(),
         "stop simulation when at least one result converges below this threshold");
   desc.add_options()("floor-max", po::value<double>(),
         "stop simulation when all results converge below this threshold");
   desc.add_options()("confidence", po::value<double>()->default_value(0.90),
         "confidence level for computing margin of error (e.g. 0.90 for 90%)");
   desc.add_options()("relative-error", po::value<double>()->default_value(0.15),
         "target error margin, as a fraction of result mean (e.g. 0.15 for ±15%)");
   desc.add_options()("absolute-error", po::value<double>(),
         "target error margin, as an absolute value (e.g. 0.1 for ±0.1); "
               "overrides relative-error if specified");
   desc.add_options()("accumulated-result", po::value<double>(),
         "target accumulated result (i.e. result mean x sample count); "
               "overrides absolute and relative error if specified");
   desc.add_options()("min-samples", po::value<int>(),
         "minimum number of samples");
   desc.add_options()("seed,s", po::value<libbase::int32u>(),
         "system initialization seed (random if not stated)");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      cout << desc << std::endl;
      return 0;
      }

   // Create estimator object and initilize cluster
   mymontecarlo estimator(vm["quiet"].as<bool>());
   switch (estimator.enable(vm["endpoint"].as<std::string>(),
         vm["quiet"].as<bool>(), vm["priority"].as<int>()))
      {
      case mymontecarlo::mode_slave:
         break;

      case mymontecarlo::mode_local:
      case mymontecarlo::mode_master:
         // If this is a server instance, check the remaining parameters
         if (vm.count("system-file") == 0 || vm.count("results-file") == 0
               || vm.count("start") == 0 || vm.count("stop") == 0
               || (vm.count("step") == 0 && vm.count("mul") == 0)
               || (vm.count("step") && vm.count("mul")))
            {
            cout << desc << std::endl;
            return 0;
            }

         // main process
            {
            // Simulation system & parameters
            estimator.set_resultsfile(vm["results-file"].as<std::string>());
            libcomm::experiment *system = createsystem(
                  vm["system-file"].as<std::string>());
            estimator.bind(system);
            libbase::vector<double> pset;
            if (vm.count("step"))
               pset = getlinrange(vm["start"].as<double>(),
                     vm["stop"].as<double>(), vm["step"].as<double>());
            else
               pset = getlogrange(vm["start"].as<double>(),
                     vm["stop"].as<double>(), vm["mul"].as<double>());
            estimator.set_confidence(vm["confidence"].as<double>());
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

            // Work out the following for every SNR value required
            for (int i = 0; i < pset.size(); i++)
               {
               system->set_parameter(pset(i));

               cerr << "Simulating system at parameter = " << pset(i)
                     << std::endl;
               libbase::vector<double> estimate, errormargin;
               estimator.estimate(estimate, errormargin);

               cerr << "Statistics: " << setprecision(4)
                     << estimator.get_samplecount() << " samples in "
                     << estimator.get_timer() << " - "
                     << estimator.get_samplecount()
                           / estimator.get_timer().elapsed() << " samples/sec"
                     << std::endl;

               // handle pre-mature breaks
               if (estimator.interrupt() && !estimator.interrupt_was_soft())
                  break;
               if (vm.count("floor-min")
                     && estimate.min() < vm["floor-min"].as<double>())
                  break;
               if (vm.count("floor-max")
                     && estimate.max() < vm["floor-max"].as<double>())
                  break;
               }

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
   return simcommsys::main(argc, argv);
   }
