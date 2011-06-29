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

/*!
 * \file
 * \brief   CUDA utilities.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "cuda-all.h"
#include "sysvar.h"

namespace cuda {

//! Get the current device

int cudaGetCurrentDevice()
   {
   int device;
   cudaSafeCall(cudaGetDevice(&device));
   return device;
   }

//! Get the number of multiprocessors for the given device

int cudaGetMultiprocessorCount(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return prop.multiProcessorCount;
   }

// Get the number of cores per multiprocessor for the given device

int cudaGetMultiprocessorSize(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   const int nGpuArchCoresPerSM[] = {-1, 8, 32};
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return nGpuArchCoresPerSM[prop.major];
   }

//! Get the warp size for the given device

int cudaGetWarpSize(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return prop.warpSize;
   }

//! Get the clock rate in GHz for the given device

double cudaGetClockRate(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return prop.clockRate * 1e-6f;
   }

//! Get the name for the given device

std::string cudaGetDeviceName(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return prop.name;
   }

//! Get the amount of global memory (in bytes) for the given device

int cudaGetGlobalMem(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return prop.totalGlobalMem;
   }

//! Get the compute capability for the given device

int cudaGetComputeCapability(int device)
   {
   if (device < 0)
      device = cudaGetCurrentDevice();
   cudaDeviceProp prop;
   cudaSafeCall(cudaGetDeviceProperties(&prop, device));
   return prop.major * 1000 + prop.minor;
   }

//! Get the number of CUDA-capable devices

int cudaGetDeviceCount()
   {
   int devices = 0;
   cudaSafeCall(::cudaGetDeviceCount(&devices));
   return devices;
   }

//! Get the version number for the device driver

int cudaGetDriverVersion()
   {
   int driverVersion = 0;
   cudaSafeCall(cudaDriverGetVersion(&driverVersion));
   return driverVersion;
   }

//! Get the version number for the CUDA runtime

int cudaGetRuntimeVersion()
   {
   int runtimeVersion = 0;
   cudaSafeCall(cudaRuntimeGetVersion(&runtimeVersion));
   return runtimeVersion;
   }

//! Format the given version into a printable string

std::string cudaPrettyVersion(int version)
   {
   std::ostringstream sout;
   sout << (version / 1000) << "." << (version % 100);
   return sout.str();
   }

// Returns the best GPU (with maximum GFLOPS)

int cudaGetMaxGflopsDeviceId()
   {
   int devices = cudaGetDeviceCount();

   int max_gflops_device = 0;
   double max_gflops = 0;

   for (int i = 0; i < devices; i++)
      {
      double gflops = cudaGetMultiprocessorCount(i)
            * cudaGetMultiprocessorSize(i) * cudaGetClockRate(i);
      if (gflops > max_gflops)
         {
         max_gflops = gflops;
         max_gflops_device = i;
         }
      }

   return max_gflops_device;
   }

//! Kernel to determine the compute capability of current device

__global__
void getcomputemodel_kernel(value_reference<int> dev_cm)
   {
#ifndef __CUDA_ARCH__
   // should be for host code path only
   const int major = 0;
   const int minor = 0;
#else
   // device code path (any architecture model)
   const int major = __CUDA_ARCH__ / 100;
   const int minor = __CUDA_ARCH__ % 100;
#endif
   // convert into the same encoding as used for the driver/runtime version
   dev_cm() = major * 1000 + minor;
   }

//! Determine the compute capability for which this code was compiled

int cudaGetComputeModel()
   {
   // allocate space for results
   value<int> dev_cm;
   dev_cm.init();
   // call the kernel
   getcomputemodel_kernel<<<1,1>>>(dev_cm);
   cudaSafeThreadSynchronize();
   // copy results back
   return dev_cm;
   }

//! Initialize the runtime and choose the best device

void cudaInitialize(std::ostream& sout)
   {
   static bool initialized = false;
   if (initialized)
      return;
   initialized = true;
   // select device to use
   int device;
   libbase::sysvar user_device("CUDA_DEVICE");
   if (user_device.is_defined())
      device = user_device.as_int();
   else
      device = cudaGetMaxGflopsDeviceId();
   cudaSafeCall(cudaSetDevice(device));
   // report to user
   sout << "CUDA device: " << device << " (" << cudaGetDeviceName() << ", "
         << (cudaGetGlobalMem() >> 20) << " MiB, "
         << cudaGetMultiprocessorSize() << "x" << cudaGetMultiprocessorCount()
         << " @ " << cudaGetClockRate() << " GHz" << ", capability "
         << cudaPrettyVersion(cudaGetComputeCapability()) << ")" << std::endl;
   sout << "CUDA initialized: compute model " << cudaPrettyVersion(
         cudaGetComputeModel()) << ", cuda runtime " << cudaPrettyVersion(
         cudaGetDriverVersion()) << std::endl;
   if (cudaGetRuntimeVersion() != cudaGetDriverVersion())
      sout << "CUDA warning: this code was compiled with cuda runtime "
            << cudaPrettyVersion(cudaGetDriverVersion()) << std::endl;
   }

//! List CUDA capable devices and their properties

void cudaQueryDevices(std::ostream& sout)
   {
   // get and report the number of CUDA capable devices
   int devices = cudaGetDeviceCount();
   if (devices == 0)
      {
      sout << "There is no device supporting CUDA" << std::endl;
      return;
      }
   else if (devices == 1)
      sout << "There is 1 device supporting CUDA" << std::endl;
   else
      sout << "There are " << devices << " devices supporting CUDA"
            << std::endl;

   // print driver and runtime versions
   sout << "  CUDA Driver Version:\t" << cudaPrettyVersion(
         cudaGetDriverVersion()) << std::endl;
   sout << "  CUDA Runtime Version:\t" << cudaPrettyVersion(
         cudaGetRuntimeVersion()) << std::endl;

   // print important details for all devices found
   for (int i = 0; i < devices; i++)
      {
      sout << std::endl;
      sout << "Device " << i << ": \"" << cudaGetDeviceName(i) << "\""
            << std::endl;
      sout << "  CUDA Capability:\t" << cudaPrettyVersion(
            cudaGetComputeCapability(i)) << std::endl;
      sout << "  Global memory:\t" << cudaGetGlobalMem(i) << " bytes"
            << std::endl;
      sout << "  Multiprocessors:\t" << cudaGetMultiprocessorCount(i)
            << std::endl;
      sout << "  Total Cores:\t" << cudaGetMultiprocessorSize(i)
            * cudaGetMultiprocessorCount(i) << std::endl;
      sout << "  Clock rate:\t" << cudaGetClockRate(i) << " GHz" << std::endl;

      // Get the properties for the given device
      cudaDeviceProp prop;
      cudaSafeCall(cudaGetDeviceProperties(&prop, i));
      sout << "  Memory per block:\t" << prop.sharedMemPerBlock << " bytes"
            << std::endl;
      sout << "  Threads per block:\t" << prop.maxThreadsPerBlock << std::endl;
      }
   }

} // end namespace
