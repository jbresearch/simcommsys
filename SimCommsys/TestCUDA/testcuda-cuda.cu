#include "testcuda.h"
#include "cuda/timer.h"
#include "cuda/vector.h"
#include <cstdio>

// *** Timing: kernel call overhead

namespace cuda {

__global__
void empty_thread()
   {
#if __CUDA_ARCH__ >= 200
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i == 0)
      {
      printf("Thread %d\n", i);
      }
#endif
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
      {
      cuda::empty_thread <<<gridsize, blocksize>>> ();
      cudaSafeWaitForKernel();
      }
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

// *** Capability test: use of classes in device code

namespace cuda {

class complex {
private:
   float r, i;
public:
   // constructor (works on host or device)
#ifdef __CUDACC__
   __device__ __host__
#endif
   complex(float r = 0.0, float i = 0.0) :
      r(r), i(i)
      {
      }
   // operators (works on host or device)
#ifdef __CUDACC__
   __device__ __host__
#endif
   complex& operator+=(const complex& x)
      {
      r += x.r;
      i += x.i;
      return *this;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   complex operator+(const complex& x)
      {
      complex y = *this;
      y += x;
      return y;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bool operator==(const complex& x) const
      {
      return r == x.r && i == x.i;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bool operator!=(const complex& x) const
      {
      return r != x.r || i != x.i;
      }
   // stream output (host only)
   friend std::ostream& operator<<(std::ostream& sout, const complex& x)
      {
      sout << x.r << " + j" << x.i;
      return sout;
      }
};

__global__
void test_useofclasses(vector_reference<complex> x)
   {
   const int N = x.size();
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < N)
      x(i) += complex(1.0, -1.0);
   }

// explicit instantiations

template class vector<complex> ;

} // end namespace

namespace testcuda {

void test_useofclasses(libbase::vector<cuda::complex>& x)
   {
   const int N = x.size();
   for (int i = 0; i < N; i++)
      x(i) += cuda::complex(1.0, -1.0);
   }

void test_useofclasses()
   {
   const int N = 5;
   std::cout << std::endl;
   std::cout << "Test use of classes in device code:" << std::endl;
   // create and fill in host vector
   libbase::vector<cuda::complex> x(N);
   for (int i = 0; i < x.size(); i++)
      x(i) = cuda::complex(i, i);
   std::cout << "Input = " << x;
   // create and copy device vector
   cuda::vector<cuda::complex> dev_x;
   dev_x = x;
   // do kernel call
   cuda::test_useofclasses<<<1, N>>> (dev_x);
   cudaSafeWaitForKernel();
   // copy results back and display
   libbase::vector<cuda::complex> y;
   y = libbase::vector<cuda::complex>(dev_x);
   std::cout << "Output (GPU) = " << y;
   // compute results on CPU and display
   test_useofclasses(x);
   std::cout << "Output (CPU) = " << x;
   // confirm results
   assert(x.isequalto(y));
   }

} // end namespace

// *** Capability test: item sizes

namespace cuda {

void get_sizes(libbase::vector<int>& result)
   {
   int i = 0;

   result(i++) = sizeof(bool);
   result(i++) = sizeof(char);
   result(i++) = sizeof(short);
   result(i++) = sizeof(int);
   result(i++) = sizeof(long);
   result(i++) = sizeof(long long);

   result(i++) = sizeof(float);
   result(i++) = sizeof(double);
   result(i++) = sizeof(long double);

   result(i++) = sizeof(void *);
   result(i++) = sizeof(cuda::vector<int>);
   result(i++) = sizeof(cuda::vector_auto<int>);
   result(i++) = sizeof(cuda::matrix<int>);
   result(i++) = sizeof(cuda::matrix_auto<int>);
   }

__device__
void get_sizes(cuda::vector_reference<int>& result)
   {
   int i = 0;

   result(i++) = sizeof(bool);
   result(i++) = sizeof(char);
   result(i++) = sizeof(short);
   result(i++) = sizeof(int);
   result(i++) = sizeof(long);
   result(i++) = sizeof(long long);

   result(i++) = sizeof(float);
   result(i++) = sizeof(double);
   result(i++) = sizeof(long double);

   result(i++) = sizeof(void *);
   result(i++) = sizeof(cuda::vector<int>);
   result(i++) = sizeof(cuda::vector_auto<int>);
   result(i++) = sizeof(cuda::matrix<int>);
   result(i++) = sizeof(cuda::matrix_auto<int>);
   }

// kernel function

__global__
void get_sizes_thread(vector_reference<int> result)
   {
   get_sizes(result);
   }

} // end namespace

namespace testcuda {

void print_sizes(libbase::vector<int>& result)
   {
   std::cout << std::endl;
   std::cout << "Type       \tSize (bytes)" << std::endl;
   std::cout << "~~~~       \t~~~~~~~~~~~~" << std::endl;

   int i = 0;
   std::cout << "bool       \t" << result(i++) << std::endl;
   std::cout << "char       \t" << result(i++) << std::endl;
   std::cout << "short      \t" << result(i++) << std::endl;
   std::cout << "int        \t" << result(i++) << std::endl;
   std::cout << "long       \t" << result(i++) << std::endl;
   std::cout << "long long  \t" << result(i++) << std::endl;
   std::cout << std::endl;
   std::cout << "float      \t" << result(i++) << std::endl;
   std::cout << "double     \t" << result(i++) << std::endl;
   std::cout << "long double\t" << result(i++) << std::endl;
   std::cout << std::endl;
   std::cout << "void *     \t" << result(i++) << std::endl;
   std::cout << "\'int\' containers:" << std::endl;
   std::cout << "vector     \t" << result(i++) << std::endl;
   std::cout << "vector_auto\t" << result(i++) << std::endl;
   std::cout << "matrix     \t" << result(i++) << std::endl;
   std::cout << "matrix_auto\t" << result(i++) << std::endl;
   }

void test_sizes()
   {
   // determine sizes on host
   libbase::vector<int> host(14);
   cuda::get_sizes(host);
   // determine sizes on gpu
   cuda::vector<int> gpu_dev;
   gpu_dev.init(14);
   cuda::get_sizes_thread <<<1,1>>> (gpu_dev);
   cudaSafeWaitForKernel();
   libbase::vector<int> gpu = libbase::vector<int>(gpu_dev);
   // print results
   std::cout << std::endl << "Sizes on host CPU" << std::endl;
   print_sizes(host);
   std::cout << std::endl << "Sizes on GPU" << std::endl;
   print_sizes(gpu);
   assert(host.isequalto(gpu));
   }

} // end namespace
