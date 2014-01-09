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

#include "testcuda.h"
#include "event_timer.h"

#include <boost/program_options.hpp>

namespace testcuda {

void time_timer()
   {
   // definitions
   libbase::cputimer t("CPU");
   libbase::event_timer<libbase::cputimer> tinner("inner timer");
   const int N = int(1e5);
   // timed loop
   t.start();
   for (int i = 0; i < N; i++)
      {
      tinner.start();
      tinner.stop();
      }
   t.stop();
   // compute and show
   const double time = t.elapsed() / N;
   std::cout << "CPU timer overhead: " << libbase::timer::format(time)
         << std::endl;
   }

/*!
 * \brief   Test program for CUDA interface
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("test,t", po::value<int>()->default_value(0),
         "test to run (0=all)");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const int t = vm["test"].as<int> ();

   if (t == 0 || t == 1)
      time_timer();
#ifdef USE_CUDA
   if (t == 0 || t == 2)
   cuda::cudaInitialize(std::cout);
   if (t == 0 || t == 3)
   cuda::cudaQueryDevices(std::cout);
   if (t == 0 || t == 4)
   cuda::time_kernelcalls();
   if (t == 0 || t == 5)
   cuda::test_useofclasses();
   if (t == 0 || t == 6)
   cuda::test_streams();
   if (t == 0 || t == 7)
   cuda::test_sizes();
#else
   failwith("CUDA support not enabled on this system");
#endif
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testcuda::main(argc, argv);
   }
