#include "testcuda.h"
#include "cuda/timer.h"

namespace cuda {

__global__
void empty_thread()
   {
   //const int i = blockIdx.x * blockDim.x + threadIdx.x;
   }

} // end namespace

namespace testcuda {

void time_kernelcalls_with(int gridsize, int blocksize)
   {
   // definitions
   cuda::timer t("GPU");
   const int N = 1e3;
   // timed loop
   t.start();
   for (int i = 0; i < N; i++)
   cuda::empty_thread <<<gridsize, blocksize>>> ();
   t.stop();
   // compute and show
   const double time = t.elapsed() / N;
   std::cout << "Kernel overhead (" << gridsize << "x" << blocksize << "): "
         << libbase::timer::format(time) << std::endl;
   }

void time_kernelcalls()
   {
   const int mpcount = cuda::cudaGetMultiprocessorCount();
   const int mpsize = cuda::cudaGetMultiprocessorSize();
   const int warp = cuda::cudaGetWarpSize();

   for (int g = 0; g <= mpcount * 4; g += mpcount)
      for (int b = 0; b <= warp; b += mpsize)
         {
         const int gridsize = g == 0 ? 1 : g;
         const int blocksize = b == 0 ? 1 : b;
         time_kernelcalls_with(gridsize, blocksize);
         }
   }

} // end namespace
