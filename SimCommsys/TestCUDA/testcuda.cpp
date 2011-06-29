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

#include "testcuda.h"
#include "event_timer.h"

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
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   time_timer();
#ifdef USE_CUDA
   cuda::cudaInitialize(std::cout);
   cuda::cudaQueryDevices(std::cout);
   time_kernelcalls();
   test_useofclasses();
   test_sizes();
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
