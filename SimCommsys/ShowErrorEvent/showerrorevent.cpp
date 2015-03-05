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

#include "config.h"
#include "randgen.h"
#include "truerand.h"
#include "serializer_libcomm.h"
#include "experiment/binomial/commsys_simulator.h"
#include "cputimer.h"

#include <boost/program_options.hpp>

#include <iostream>

namespace showerrorevent {

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

using std::cout;
using std::cerr;
namespace po = boost::program_options;

libcomm::experiment *createsystem(const std::string& fname)
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;
   // load system from string representation
   libcomm::experiment *system;
   std::ifstream file(fname.c_str(), std::ios_base::in | std::ios_base::binary);
   file >> system >> libbase::verifycomplete;
   return system;
   }

/*! \brief Seed the random generators in the experiment
 *
 * Use the given seed to initialize a PRNG for seeding the embedded system.
 */
void seed_experiment(libcomm::experiment *system, const libbase::int32u seed)
   {
   libbase::randgen prng;
   prng.seed(seed);
   system->seedfrom(prng);
   cerr << "Seed: " << seed << std::endl;
   }

/*! \brief Randomly seed the random generators in the experiment
 *
 * Use a true RNG to determine the initial seed value, and seed the experiment.
 */
void seed_experiment(libcomm::experiment *system)
   {
   libbase::truerand trng;
   libbase::int32u seed = trng.ival();
   seed_experiment(system, seed);
   }

void display_event(libcomm::experiment *system)
   {
   libbase::vector<int> last_event = system->get_event();
   const int tau = last_event.size() / 2;
   for (int i = 0; i < tau; i++)
      {
      cout << i << '\t' << last_event(i) << '\t' << last_event(i + tau);
      if (last_event(i) != last_event(i + tau))
         cout << "\t*";
      cout << std::endl;
      }
   }

/*!
 * \brief   Error Event Analysis for Experiments
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("system-file,i", po::value<std::string>(),
         "input file containing system description");
   desc.add_options()("parameter,r", po::value<double>(),
         "simulation parameter");
   desc.add_options()("seed,s", po::value<libbase::int32u>(),
         "system initialization seed (random if not stated)");
   desc.add_options()("show-all,a", po::bool_switch(),
         "show all simulated frames (not just first error event)");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("system-file") == 0
         || vm.count("parameter") == 0)
      {
      cout << desc << std::endl;
      return 0;
      }

   // Interpret user parameters
   const bool showall = vm["show-all"].as<bool> ();
   // Simulation system & parameters
   libcomm::experiment *system = createsystem(
         vm["system-file"].as<std::string> ());
   system->set_parameter(vm["parameter"].as<double> ());
   // Initialise running values
   system->reset();
   if (vm.count("seed"))
      seed_experiment(system, vm["seed"].as<libbase::int32u>());
   else
      seed_experiment(system);
   cerr << "Simulating system at parameter = " << system->get_parameter()
         << std::endl;
   // Simulate, waiting for an error event
   libbase::vector<double> result;
   do
      {
      cerr << "Simulating sample " << system->get_samplecount() << std::endl;
      system->sample(result);
      system->accumulate(result);
      if (showall)
         {
         cerr << "Event for sample " << system->get_samplecount() << ":"
               << std::endl;
         display_event(system);
         }
      } while (result.min() == 0);
   cerr << "Event found after " << system->get_samplecount() << " samples"
         << std::endl;
   // Display results if necessary
   if (!showall)
      display_event(system);

   // Destroy what was created on the heap
   delete system;
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return showerrorevent::main(argc, argv);
   }
