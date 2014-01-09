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

#include "mapper/map_aggregating.h"
#include "randgen.h"
#include "vectorutils.h"
#include "vector_itfunc.h"

#include <boost/program_options.hpp>

#include <iostream>

namespace testmapper {

using std::cout;
using std::cerr;
using libbase::vector;
namespace po = boost::program_options;

// Type definitions
typedef vector<double> array1d_t;
typedef vector<array1d_t> array1vd_t;

void test_aggregating(int N, int q, int M)
   {
   // define mapper according to specifications
   libcomm::map_aggregating<vector, double, double> mapper;
   libbase::randgen prng;
   prng.seed(0);
   mapper.seedfrom(prng);
   mapper.set_parameters(q, M);
   mapper.set_blocksize(libbase::size_type<vector>(N));
   // determine block size at blockmodem input
   const int tau = mapper.output_block_size();
   // allocate space for blockmodem posteriors
   array1vd_t ptable_modem_post;
   libbase::allocate(ptable_modem_post, tau, M);
   // create a fake set of normalized probabilities
   for (int i = 0; i < tau; i++)
      for (int d = 0; d < M; d++)
         ptable_modem_post(i)(d) = prng.fval_halfopen();
   libbase::normalize_results(ptable_modem_post, ptable_modem_post);
   // inverse mapping to obtain decoder input
   array1vd_t ptable_codec_prior;
   mapper.inverse(ptable_modem_post, ptable_codec_prior);
   // forward mapping to obtain mapper input
   array1vd_t ptable_modem_prior;
   mapper.transform(ptable_codec_prior, ptable_modem_prior);
   // final check
   bool identical = true;
   for (int i = 0; i < tau && identical; i++)
      identical = ptable_modem_post(i).isequalto(ptable_modem_prior(i));
   // final output
   if (identical)
      cout << "Vectors identical" << std::endl;
   else
      {
      cout.setf(std::ios::fixed, std::ios::floatfield);
      cout.precision(6);
      cout << "Modem probabilities:" << std::endl;
      for (int i = 0; i < tau; i++)
         {
         cout << "Index " << i << ":" << std::endl;
         cout << "\tPost:\t";
         ptable_modem_post(i).serialize(cout);
         cout << "\tPrior:\t";
         ptable_modem_prior(i).serialize(cout);
         }
      cout << "Codec probabilities:" << std::endl;
      for (int i = 0; i < N; i++)
         {
         cout << "Index " << i << ":\t";
         ptable_codec_prior(i).serialize(cout);
         }
      }
   }

/*!
 * \brief   Test program for non-trivial mappers
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("blocksize,N", po::value<int>(),
         "block size in symbols at encoder output");
   desc.add_options()("q,q", po::value<int>(),
         "alphabet size for encoder output");
   desc.add_options()("M,M", po::value<int>(),
         "alphabet size for blockmodem input");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("blocksize") == 0 || vm.count("q") == 0
         || vm.count("M") == 0)
      {
      cout << desc << std::endl;
      return 0;
      }

   // Shorthand for user parameters
   const int N = vm["blocksize"].as<int>();
   const int q = vm["q"].as<int>();
   const int M = vm["M"].as<int>();

   // Run tests
   test_aggregating(N, q, M);

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testmapper::main(argc, argv);
   }
