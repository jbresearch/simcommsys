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