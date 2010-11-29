#include "channel/bsid2d.h"
#include "randgen.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace testbsid2d {

using std::cout;
using std::cerr;
using libbase::matrix;
using libbase::randgen;
using libcomm::bsid2d;

void visualtest(int seed, int type, double p)
   {
   // define an alternating input sequence
   const int M = 5, N = 5;
   matrix<bool> tx(M, N);
   switch (type)
      {
      case 0:
      case 1:
         tx = type != 0;
         break;
      case 2:
         for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
               tx(i, j) = ((i & 1) ^ (j & 1)) != 0;
         break;
      default:
         failwith("Invalid type");
      }
   cout << "Tx: " << tx << std::endl;
   // pass that through the channel
   matrix<bool> rx1, rx2;
   // seed generator
   randgen prng;
   prng.seed(seed);
   // channel1 is a substitution-only channel
   bsid2d channel1(true, false, false);
   channel1.seedfrom(prng);
   channel1.set_parameter(p);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << std::endl;
   // channel1 is an insdel-only channel
   bsid2d channel2(false, true, true);
   channel2.seedfrom(prng);
   channel2.set_parameter(p);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << std::endl;
   }

/*!
 * \brief   Test program for 2D BSID channel
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("seed,s", po::value<int>()->default_value(0),
         "random generator seed");
   desc.add_options()("parameter,p", po::value<double>()->default_value(0.1),
         "channel error probability");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << std::endl;
      return 1;
      }

   // Get user parameters
   const int seed = vm["seed"].as<int> ();
   const double p = vm["parameter"].as<double> ();

   // create a test sequence and test 2D BSID transmission
   visualtest(seed, 0, p); // all-zero sequence
   visualtest(seed, 1, p); // all-one sequenceee
   visualtest(seed, 2, p); // random sequence

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testbsid2d::main(argc, argv);
   }
