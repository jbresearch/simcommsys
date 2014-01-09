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

#include <iostream>
#include <cstdlib>

// error wrappers

#define cudaSafeCall(error) __cudaSafeCall(error, __FILE__, __LINE__)

inline void __cudaSafeCall(const cudaError_t error, const char *file,
      const int line)
   {
   if (error == cudaSuccess)
      return;
   std::cerr << "CUDA error in file <" << file << ">, line " << line << " : "
         << cudaGetErrorString(error) << ".\n";
   cudaThreadExit();
   exit(1);
   }

// main program

int main()
   {
   int devices = 0;
   cudaSafeCall(cudaGetDeviceCount(&devices));

   // check we have at least one device installed
   if (devices == 0)
      {
      std::cerr << "No CUDA-capable devices found" << std::endl;
      exit(1);
      }

   // determine highest compute capability for installed devices
   int max_cc = 0;
   for (int i = 0; i < devices; i++)
      {
      cudaDeviceProp prop;
      cudaSafeCall(cudaGetDeviceProperties(&prop, i));
      const int cc = prop.major * 10 + prop.minor;
      max_cc = std::max(max_cc, cc);
      }

   // print the architecture number
   std::cout << max_cc << std::endl;

   return 0;
   }
