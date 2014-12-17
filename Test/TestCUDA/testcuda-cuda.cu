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
#include "cuda/gputimer.h"
#include "cuda/stream.h"
#include "cuda/vector.h"
#include "cuda/value.h"
#include <cstdio>

#include "algorithm/fba2-cuda.h"
#include "modem/tvb-receiver-cuda.h"
#include "channel/qids.h"

// *** Timing: kernel call overhead

namespace cuda {

__global__
void empty_kernel(bool show)
   {
#if __CUDA_ARCH__ >= 200
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (show && i == 0)
      {
      printf("Thread %d/%d of Block %d/%d\n", threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
      }
#endif
   }

void time_kernelcalls_with(int gridsize, int blocksize)
   {
   // definitions
   gputimer t("GPU");
   const int N = 1e3;
   // timed loop
   t.start();
   for (int i = 0; i < N; i++)
      {
empty_kernel      <<<gridsize, blocksize>>> (i == 0);
      }
   t.stop();
   // compute and show
   const double time = t.elapsed() / N;
   std::cout << "Kernel overhead (" << gridsize << "×" << blocksize << "): "
   << libbase::timer::format(time) << std::endl;
   }

void time_kernelcalls()
   {
   const int mpcount = cudaGetMultiprocessorCount();
   const int mpsize = cudaGetMultiprocessorSize();
   const int warp = cudaGetWarpSize();

   for (int g = 0; g <= mpcount * 4; g += mpcount)
   for (int b = 0; b <= warp; b += mpsize)
      {
      const int gridsize = g == 0 ? 1 : g;
      const int blocksize = b == 0 ? 1 : b;
      time_kernelcalls_with(gridsize, blocksize);
      }
   }

// *** Capability test: use of classes in device code

class complex
   {
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
void test_useofclasses_kernel(vector_reference<complex> x)
   {
   const int N = x.size();
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < N)
   x(i) += complex(1.0, -1.0);
   }

// explicit instantiations

template class vector<complex>;

void test_useofclasses(libbase::vector<complex>& x)
   {
   const int N = x.size();
   for (int i = 0; i < N; i++)
   x(i) += complex(1.0, -1.0);
   }

void test_useofclasses()
   {
   const int N = 5;
   std::cout << std::endl;
   std::cout << "Test use of classes in device code:" << std::endl;
   // create and fill in host vector
   libbase::vector<cuda::complex> x(N);
   for (int i = 0; i < x.size(); i++)
   x(i) = complex(i, i);
   std::cout << "Input = " << x;
   // create and copy device vector
   vector<complex> dev_x;
   dev_x = x;
   // do kernel call
   test_useofclasses_kernel<<<1, N>>> (dev_x);
   cudaSafeDeviceSynchronize();
   // copy results back and display
   libbase::vector<complex> y;
   y = libbase::vector<complex>(dev_x);
   std::cout << "Output (GPU) = " << y;
   // compute results on CPU and display
   test_useofclasses(x);
   std::cout << "Output (CPU) = " << x;
   // confirm results
   assert(x.isequalto(y));
   }

// *** Capability test: parallel execution of streams

__global__
void test_streams_kernel(value_reference<double> result)
   {
   const int N = 1e5;
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   // waste some time
   double r = 0;
   for (int j=1; j<N; j+=4)
   r += 4.0 / j - 4.0 / (j+2);
   // return result
   result() = r;
   }

void test_streams()
   {
   std::cout << std::endl;
   std::cout << "Test parallel execution of streams:" << std::endl;
   // space for stream data
   const int N = 2 * cudaGetMultiprocessorCount();
   stream vs[N];
   gputimer vt[N];
   value<double> vr[N];
   for (int i=0; i<N; i++)
      {
      vt[i].set_stream(vs[i].get_id());
      vr[i].init();
      }
   // call the two kernels in parallel as separate streams
   libbase::cputimer tcpu("sequence");
   gputimer tgpu("sequence");
   for (int i=0; i<N; i++)
      {
      vt[i].start();
      test_streams_kernel<<<1,1,0,vs[i].get_id()>>>(vr[i]);
      }
   // delay issue of stream stop timer, to avoid breaking concurrency
   for (int i=0; i<N; i++)
      {
      vt[i].stop();
      }
   tgpu.stop();
   // wait for all CUDA events to finish before stopping CPU timer
   cudaSafeDeviceSynchronize();
   tcpu.stop();
   // show what happened
   for (int i=0; i<N; i++)
      {
      std::cout << "Kernel " << i << ": " << vt[i] << "\t";
      std::cout << "Result = " << vr[i] << std::endl;
      }
   std::cout << "Kernel sequence: " << tgpu << std::endl;
   std::cout << "Kernel sequence (CPU): " << tcpu << std::endl;
   }

// *** Capability test: item sizes

void get_descriptors(libbase::vector<std::string>& names)
   {
   int i = 0;

   names(i++) = "bool";
   names(i++) = "char";
   names(i++) = "short";
   names(i++) = "int";
   names(i++) = "long";
   names(i++) = "long long";

   names(i++) = "float";
   names(i++) = "double";
   names(i++) = "long double";

   names(i++) = "void*";
   names(i++) = "size_t";

   names(i++) = "vector<int>";
   names(i++) = "vector_auto<int>";
   names(i++) = "matrix<int>";
   names(i++) = "matrix_auto<int>";

   names(i++) = "libcomm::qids<bool,float>::metric_computer";
   names(i++) = "tvb_receiver<bool,float,float>";
   names(i++) = "tvb_receiver<bool,double,float>";
   names(i++) = "fba2<tvb_receiver,bool,float,float>::metric_computer";
   names(i++) = "fba2<tvb_receiver,bool,double,float>::metric_computer";
   names(i++) = "fba2<tvb_receiver,bool,float,float>";
   names(i++) = "fba2<tvb_receiver,bool,double,float>";
   }

void get_sizes(libbase::vector<int>& sizes, libbase::vector<int>& align)
   {
   int i;

   // sizes
   i = 0;
   sizes(i++) = sizeof(bool);
   sizes(i++) = sizeof(char);
   sizes(i++) = sizeof(short);
   sizes(i++) = sizeof(int);
   sizes(i++) = sizeof(long);
   sizes(i++) = sizeof(long long);

   sizes(i++) = sizeof(float);
   sizes(i++) = sizeof(double);
   sizes(i++) = sizeof(long double);

   sizes(i++) = sizeof(void *);
   sizes(i++) = sizeof(size_t);

   sizes(i++) = sizeof(vector<int>);
   sizes(i++) = sizeof(vector_auto<int>);
   sizes(i++) = sizeof(matrix<int>);
   sizes(i++) = sizeof(matrix_auto<int>);

   sizes(i++) = sizeof(libcomm::qids<bool,float>::metric_computer);
   sizes(i++) = sizeof(tvb_receiver<bool,float,float>);
   sizes(i++) = sizeof(tvb_receiver<bool,double,float>);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>::metric_computer);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>::metric_computer);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>);

   // alignment
   i = 0;
   align(i++) = __alignof__(bool);
   align(i++) = __alignof__(char);
   align(i++) = __alignof__(short);
   align(i++) = __alignof__(int);
   align(i++) = __alignof__(long);
   align(i++) = __alignof__(long long);

   align(i++) = __alignof__(float);
   align(i++) = __alignof__(double);
   align(i++) = __alignof__(long double);

   align(i++) = __alignof__(void *);
   align(i++) = __alignof__(size_t);

   align(i++) = __alignof__(vector<int>);
   align(i++) = __alignof__(vector_auto<int>);
   align(i++) = __alignof__(matrix<int>);
   align(i++) = __alignof__(matrix_auto<int>);

   align(i++) = __alignof__(libcomm::qids<bool,float>::metric_computer);
   align(i++) = __alignof__(tvb_receiver<bool,float,float>);
   align(i++) = __alignof__(tvb_receiver<bool,double,float>);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>::metric_computer);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>::metric_computer);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>);
   }

__device__
void get_sizes(vector_reference<int>& sizes, vector_reference<int>& align)
   {
   int i;

   // sizes
   i = 0;
   sizes(i++) = sizeof(bool);
   sizes(i++) = sizeof(char);
   sizes(i++) = sizeof(short);
   sizes(i++) = sizeof(int);
   sizes(i++) = sizeof(long);
   sizes(i++) = sizeof(long long);

   sizes(i++) = sizeof(float);
   sizes(i++) = sizeof(double);
   sizes(i++) = sizeof(long double);

   sizes(i++) = sizeof(void *);
   sizes(i++) = sizeof(size_t);

   sizes(i++) = sizeof(vector<int>);
   sizes(i++) = sizeof(vector_auto<int>);
   sizes(i++) = sizeof(matrix<int>);
   sizes(i++) = sizeof(matrix_auto<int>);

   sizes(i++) = sizeof(libcomm::qids<bool,float>::metric_computer);
   sizes(i++) = sizeof(tvb_receiver<bool,float,float>);
   sizes(i++) = sizeof(tvb_receiver<bool,double,float>);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>::metric_computer);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>::metric_computer);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>);
   sizes(i++) = sizeof(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>);

   // alignment
   i = 0;
   align(i++) = __alignof__(bool);
   align(i++) = __alignof__(char);
   align(i++) = __alignof__(short);
   align(i++) = __alignof__(int);
   align(i++) = __alignof__(long);
   align(i++) = __alignof__(long long);

   align(i++) = __alignof__(float);
   align(i++) = __alignof__(double);
   align(i++) = __alignof__(long double);

   align(i++) = __alignof__(void *);
   align(i++) = __alignof__(size_t);

   align(i++) = __alignof__(vector<int>);
   align(i++) = __alignof__(vector_auto<int>);
   align(i++) = __alignof__(matrix<int>);
   align(i++) = __alignof__(matrix_auto<int>);

   align(i++) = __alignof__(libcomm::qids<bool,float>::metric_computer);
   align(i++) = __alignof__(tvb_receiver<bool,float,float>);
   align(i++) = __alignof__(tvb_receiver<bool,double,float>);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>::metric_computer);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>::metric_computer);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,float,float>, bool, float, float, false, false, true>);
   align(i++) = __alignof__(fba2<tvb_receiver<bool,double,float>, bool, double, float, false, false, true>);
   }

// kernel function

__global__
void get_sizes_kernel(vector_reference<int> sizes, vector_reference<int> align)
   {
   get_sizes(sizes, align);
   }

void print_sizes(libbase::vector<std::string>& names,
      libbase::vector<int>& host, libbase::vector<int>& gpu)
   {
   std::cout << "Host\tGPU\tType" << std::endl;
   std::cout << "~~~~\t~~~\t~~~~" << std::endl;

   for (int i = 0; i < names.size(); i++)
   std::cout << host(i) << "\t" << gpu(i) << "\t" << names(i) << std::endl;
   }

void test_sizes()
   {
   const int N = 22;
   // get descriptors
   libbase::vector<std::string> names(N);
   get_descriptors(names);
   // determine sizes on host
   libbase::vector<int> sizes_h(N);
   libbase::vector<int> align_h(N);
   get_sizes(sizes_h, align_h);
   // determine sizes on gpu
   vector<int> sizes_g_d;
   vector<int> align_g_d;
   sizes_g_d.init(N);
   align_g_d.init(N);
   get_sizes_kernel<<<1,1>>> (sizes_g_d, align_g_d);
   cudaSafeDeviceSynchronize();
   libbase::vector<int> sizes_g = libbase::vector<int>(sizes_g_d);
   libbase::vector<int> align_g = libbase::vector<int>(align_g_d);
   // print results
   std::cout << std::endl;
   std::cout << "Type/object size comparison (in bytes)" << std::endl;
   std::cout << std::endl;
   print_sizes(names, sizes_h, sizes_g);
   std::cout << std::endl;
   std::cout << "Type/object alignment comparison (in bytes)" << std::endl;
   std::cout << std::endl;
   print_sizes(names, align_h, align_g);
   // debug checks
   assert(sizes_h.isequalto(sizes_g));
   assert(align_h.isequalto(align_g));
   }

} // end namespace
