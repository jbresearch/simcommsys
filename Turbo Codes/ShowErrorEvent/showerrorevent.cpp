#include "config.h"
#include "randgen.h"
#include "truerand.h"
#include "serializer_libcomm.h"
#include "commsys_simulator.h"
#include "timer.h"

#include <boost/program_options.hpp>

#include <iostream>

using std::cout;
using std::cerr;
namespace po = boost::program_options;

libcomm::experiment *createsystem(const std::string& fname)
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;
   // load system from string representation
   libcomm::experiment *system;
   std::ifstream file(fname.c_str(), std::ios_base::in | std::ios_base::binary);
   file >> system;
   // check for errors in loading system
   libbase::verifycompleteload(file);
   return system;
   }

void seed_experiment(libcomm::experiment *system)
   {
   libbase::truerand trng;
   libbase::randgen prng;
   const libbase::int32u seed = trng.ival();
   prng.seed(seed);
   system->seedfrom(prng);
   cerr << "Seed: " << seed << "\n";
   }

/*!
   \brief   Error Event Analysis for Experiments
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()
      ("help", "print this help message")
      ("system-file,i", po::value<std::string>(),
         "input file containing system description")
      ("parameter,p", po::value<double>(),
         "simulation parameter")
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

   // Simulation system & parameters
   libcomm::experiment *system = createsystem(vm["system-file"].as<std::string>());
   system->set_parameter(vm["parameter"].as<double>());

   // Initialise running values
   system->reset();
   seed_experiment(system);
   cerr << "Simulating system at parameter = " << system->get_parameter() << "\n";
   // Simulate, waiting for an error event
   libbase::vector<double> result;
   do {
      cerr << "Simulating sample " << system->get_samplecount() << "\n";
      system->sample(result);
      system->accumulate(result);
      } while(result.min() == 0);
   cerr << "Event found after " << system->get_samplecount() << " samples\n";
   // Display results
   libbase::vector<int> last_event = system->get_event();
   const int tau = last_event.size()/2;
   for(int i=0; i<tau; i++)
      cout << last_event(i) << '\t' << last_event(i+tau) << '\n';

   return 0;
   }
