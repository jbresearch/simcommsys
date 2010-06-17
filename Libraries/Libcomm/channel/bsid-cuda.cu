/*!
 * \file
 * \brief   Parallel code for BSID channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "bsid-cuda.h"

namespace cuda {

#define cutilSafeCall(err)  __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
   {
   if (cudaSuccess != err)
      {
      std::cerr << "cudaSafeCall() Runtime API error in file <" << file
            << ">, line " << line << " : " << cudaGetErrorString(err) << ".\n";
      exit(-1);
      }
   }

void query_devices(std::ostream& sout)
   {
   // get and report the number of CUDA capable devices
   int devices;
   cutilSafeCall(cudaGetDeviceCount(&devices));
   if (devices == 0)
      {
      sout << "There is no device supporting CUDA\n";
      return;
      }
   else if (devices == 1)
      sout << "There is 1 device supporting CUDA\n";
   else
      sout << "There are " << devices << " devices supporting CUDA\n";

   // print driver and runtime versions
   int driverVersion = 0, runtimeVersion = 0;
   cutilSafeCall(cudaDriverGetVersion(&driverVersion));
   cutilSafeCall(cudaRuntimeGetVersion(&runtimeVersion));
   sout << "  CUDA Driver Version:\t" << driverVersion / 1000 << "."
         << driverVersion % 100 << "\n";
   sout << "  CUDA Runtime Version:\t" << runtimeVersion / 1000 << "."
         << runtimeVersion % 100 << "\n";

   // print important details for all devices found
   for (int i = 0; i < devices; i++)
      {
      cudaDeviceProp prop;
      cutilSafeCall(cudaGetDeviceProperties(&prop, i));
      sout << "\nDevice " << i << ": \"" << prop.name << "\"\n";

      sout << "  CUDA Capability:\t" << prop.major << "." << prop.minor << "\n";
      sout << "  Global memory:\t" << prop.totalGlobalMem << " bytes\n";
      sout << "  Multiprocessors:\t" << prop.multiProcessorCount << "\n";
      sout << "  Total Cores:\t" << 8 * prop.multiProcessorCount << "\n";
      sout << "  Memory per block:\t" << prop.sharedMemPerBlock << " bytes\n";
      sout << "  Threads per block:\t" << prop.maxThreadsPerBlock << "\n";
      sout << "  Clock rate:\t" << prop.clockRate * 1e-6f << " GHz\n";
      }
   }

// value in device memory

template <class T>
value<T>::value(const value<T>& x)
   {
   if (x.data)
      {
      // allocate memory  
      allocate();
      // copy data from device to device
      cutilSafeCall(cudaMemcpy(base::data, x.data, sizeof(T), cudaMemcpyDeviceToDevice));
      }
   }

template <class T>
value<T>& value<T>::operator=(const value<T>& x)
   {
   if (x.data == NULL)
      {
      // reset values
      base::data = NULL;
      }
   else
      {
      // allocate memory  
      allocate();
      // copy data from device to device
      cutilSafeCall(cudaMemcpy(base::data, x.data, sizeof(T), cudaMemcpyDeviceToDevice));
      }

   return *this;
   }

template <class T>
void value<T>::allocate()
   {
   // allocate value in device memory
   void *p;
   cutilSafeCall(cudaMalloc(&p, sizeof(T)));
   base::data = (T*) p;
   }

template <class T>
void value<T>::free()
   {
   assert(base::data != NULL);
   // free device memory
   cutilSafeCall(cudaFree(base::data));
   // reset variables
   base::data = NULL;
   }

template <class T>
value<T>& value<T>::operator=(const T& x)
   {
   // allocate memory if needed  
   if (!base::data)
      allocate();

   // copy data from host to device
   cutilSafeCall(cudaMemcpy(base::data, &x, sizeof(T), cudaMemcpyHostToDevice));

   return *this;
   }

template <class T>
value<T>::operator T()
   {
   T x;

   // copy data from device to host
   cutilSafeCall(cudaMemcpy(&x, base::data, sizeof(T), cudaMemcpyDeviceToHost));

   return x;
   }

/*!
 * \brief   A single value in device memory - reference
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class value_ref : public value_interface<T> {
private:
   typedef value_interface<T> base;
public:
   /*! \brief principal constructor
    * \note This converter is a host function
    */
   value_ref(const base& x) :
      base(x)
      {
      }
   // use constructor-built copy constructor and assignment
   // no need for destructor
   /*! \brief Index operator (write-access)
    * \note Performs boundary checking.
    */
   __device__
   T& operator()()
      {
      //assert(base::data);
      return *base::data;
      }
   /*! \brief Index operator (read-only access)
    * \note Performs boundary checking.
    */
   __device__
   const T& operator()() const
      {
      //assert(base::data);
      return *base::data;
      }
};

// explicit instantiations

template class value_interface<float> ;
template class value<float> ;
template class value_ref<float> ;

// vector in device memory

template <class T>
vector<T>::vector(const vector<T>& x)
   {
   if (x.data)
      {
      // allocate memory  
      allocate(x.length);
      // copy data from device to device
      cutilSafeCall(cudaMemcpy(base::data, x.data, base::length * sizeof(T),
                  cudaMemcpyDeviceToDevice));
      }
   }

template <class T>
vector<T>& vector<T>::operator=(const vector<T>& x)
   {
   if (x.data == NULL)
      {
      // reset values
      base::data = NULL;
      base::length = 0;
      }
   else
      {
      // allocate memory  
      allocate(x.length);
      // copy data from device to device
      cutilSafeCall(cudaMemcpy(base::data, x.data, base::length * sizeof(T),
                  cudaMemcpyDeviceToDevice));
      }

   return *this;
   }

template <class T>
void vector<T>::allocate(int n)
   {
   assert(n > 0);
   base::length = n;
   // allocate vector in device memory
   void *p;
   cutilSafeCall(cudaMalloc(&p, base::length * sizeof(T)));
   base::data = (T*) p;
   }

template <class T>
void vector<T>::free()
   {
   assert(base::data != NULL);
   // free device memory
   cutilSafeCall(cudaFree(base::data));
   // reset variables
   base::data = NULL;
   base::length = 0;
   }

template <class T>
vector<T>& vector<T>::operator=(const libbase::vector<T>& x)
   {
   assert(x.size() > 0);

   // (re-)allocate memory if needed  
   if (!base::data)
      allocate(x.size());
   else if (base::length != x.size())
      {
      free();
      allocate(x.size());
      }

   // copy data from host to device
   cutilSafeCall(cudaMemcpy(base::data, &x(0), base::length * sizeof(T),
               cudaMemcpyHostToDevice));

   return *this;
   }

template <class T>
vector<T>::operator libbase::vector<T>()
   {
   libbase::vector<T> x(base::length);

   // copy data from device to host
   cutilSafeCall(cudaMemcpy(&x(0), base::data, base::length * sizeof(T),
               cudaMemcpyDeviceToHost));

   return x;
   }

/*!
 * \brief   A one-dimensional array in device memory - reference
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class vector_ref : public vector_interface<T> {
private:
   typedef vector_interface<T> base;
public:
   /*! \brief principal constructor
    * \note This converter is a host function
    */
   vector_ref(const base& x) :
      base(x)
      {
      }
   // use constructor-built copy constructor and assignment
   // no need for destructor
   /*! \brief Index operator (write-access)
    * \note Performs boundary checking.
    */
   __device__
   T& operator()(const int x)
      {
      //assert(x >= 0 && x < length);
      return base::data[x];
      }
   /*! \brief Index operator (read-only access)
    * \note Performs boundary checking.
    */
   __device__
   const T& operator()(const int x) const
      {
      //assert(x >= 0 && x < length);
      return base::data[x];
      }
   //! Total number of elements
   __device__
   int size() const
      {
      return base::length;
      }
};

// explicit instantiations

template class vector_interface<bool> ;
template class vector<bool> ;
template class vector_ref<bool> ;

template class vector_interface<float> ;
template class vector<float> ;
template class vector_ref<float> ;

// matrix in device memory

template <class T>
matrix<T>::matrix(const matrix<T>& x)
   {
   if (x.data)
      {
      // allocate memory  
      allocate(x.rows, x.cols);
      // copy data from device to device
      cutilSafeCall(cudaMemcpy(base::data, x.data, base::rows * base::pitch * sizeof(T),
                  cudaMemcpyDeviceToDevice));
      }
   }

template <class T>
matrix<T>& matrix<T>::operator=(const matrix<T>& x)
   {
   if (x.data == NULL)
      {
      // reset values
      base::data = NULL;
      base::rows = 0;
      base::cols = 0;
      base::pitch = 0;
      }
   else
      {
      // allocate memory  
      allocate(x.rows, x.cols);
      // copy data from device to device
      cutilSafeCall(cudaMemcpy(base::data, x.data, base::rows * base::pitch * sizeof(T),
                  cudaMemcpyDeviceToDevice));
      }

   return *this;
   }

template <class T>
void matrix<T>::allocate(int m, int n)
   {
   assert(m > 0);
   assert(n > 0);
   base::rows = m;
   base::cols = n;
   // allocate matrix in device memory
   void *p;
   size_t pitch;
   cutilSafeCall(cudaMallocPitch(&p, &pitch, base::cols * sizeof(T), base::rows));
   base::data = (T*) p;
   base::pitch = pitch / sizeof(T);
   }

template <class T>
void matrix<T>::free()
   {
   assert(base::data != NULL);
   // free device memory
   cutilSafeCall(cudaFree(base::data));
   // reset variables
   base::data = NULL;
   base::rows = 0;
   base::cols = 0;
   base::pitch = 0;
   }

template <class T>
matrix<T>& matrix<T>::operator=(const libbase::matrix<T>& x)
   {
   assert(x.size().rows() > 0);
   assert(x.size().cols() > 0);

   // (re-)allocate memory if needed  
   if (!base::data)
      allocate(x.size().rows(), x.size().cols());
   else if (base::rows != x.size().rows() || base::cols != x.size().cols())
      {
      free();
      allocate(x.size().rows(), x.size().cols());
      }

   // copy data from host to device
   for (int i = 0; i < base::rows; i++)
      cutilSafeCall(cudaMemcpy(&base::data[i * base::pitch], &x(i, 0),
                  base::cols * sizeof(T), cudaMemcpyHostToDevice));

   return *this;
   }

template <class T>
matrix<T>::operator libbase::matrix<T>()
   {
   libbase::matrix<T> x(base::rows, base::cols);

   // copy data from device to host
   for (int i = 0; i < base::rows; i++)
      cutilSafeCall(cudaMemcpy(&x(i, 0), &base::data[i * base::pitch],
                  base::cols * sizeof(T), cudaMemcpyDeviceToHost));

   return x;
   }

/*!
 * \brief   A two-dimensional array in device memory - reference
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class matrix_ref : public matrix_interface<T> {
private:
   typedef matrix_interface<T> base;
public:
   /*! \brief principal constructor
    * \note This converter is a host function
    */
   matrix_ref(const base& x) :
      base(x)
      {
      }
   // use constructor-built copy constructor and assignment
   // no need for destructor
   /*! \brief Index operator (write-access)
    * \note Performs boundary checking.
    */
   __device__
   T& operator()(const int i, const int j)
      {
      //assert(i >= 0 && i < rows);
      //assert(j >= 0 && j < cols);
      return base::data[i * base::pitch + j];
      }
   /*! \brief Index operator (read-only access)
    * \note Performs boundary checking.
    */
   __device__
   const T& operator()(const int i, const int j) const
      {
      //assert(i >= 0 && i < rows);
      //assert(j >= 0 && j < cols);
      return base::data[i * base::pitch + j];
      }
};

// explicit instantiations

template class matrix_interface<float> ;
template class matrix<float> ;
template class matrix_ref<float> ;

// device functions for min/max

template <class T>
__device__
inline const T& min(const T& a, const T&b)
   {
   if (a < b)
   return a;
   return b;
   }

template <class T>
__device__
inline const T& max(const T& a, const T&b)
   {
   if (a > b)
   return a;
   return b;
   }

template <class T>
__device__
inline void swap(T& a, T&b)
   {
   T tmp = a;
   a = b;
   b = tmp;
   }

// BSID-related functions

__global__ void bsid_receive_thread(value_ref<float> gl_result,
      const vector_ref<bool> tx, const vector_ref<bool> rx, const matrix_ref<
            float> Rtable, const value_ref<float> Rval, const int I,
      const int xmax, const int N)
   {
   //const int i = blockIdx.x * blockDim.x + threadIdx.x;
   typedef float real;
   // dynamically allocated float: two arrays each [2 * xmax + 1];
   extern __shared__ float F[];
   // Compute sizes
   const int n = tx.size();
   const int mu = rx.size() - n;
   // Set up two slices of forward matrix, and associated pointers
   real *Fthis = &F[0];
   real *Fprev = &F[2 * xmax + 1];
   // for prior list, reset all elements to zero
   for (int x = 0; x < 2 * xmax + 1; x++)
      Fprev[x] = 0;
   // we also know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   Fprev[xmax + 0] = 1;
   // compute remaining matrix values
   for (int j = 1; j < n; ++j)
      {
      // for this list, reset all elements to zero
      for (int x = 0; x < 2 * xmax + 1; x++)
         Fthis[x] = 0;
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y < rx.size()
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      // note: a and y are offset by xmax
      const int ymin = max(0, xmax - j);
      const int ymax = min(2 * xmax, xmax + rx.size() - j);
      for (int y = ymin; y <= ymax; ++y)
         {
         const int amin = max(max(0, xmax + 1 - j), y - I);
         const int amax = min(2 * xmax, y + 1);
         // check if the last element is a pure deletion
         int amax_act = amax;
         if (y - amax < 0)
            {
            Fthis[y] += Fprev[amax] * Rval();
            amax_act--;
            }
         // elements requiring comparison of tx and rx bits
         for (int a = amin; a <= amax_act; ++a)
            {
            // received subsequence has
            // start:  j-1+a
            // length: y-a+1
            // therefore last element is: start+length-1 = j+y-1
            const bool cmp = tx(j - 1) != rx(j + (y - xmax) - 1);
            Fthis[y] += Fprev[a] * Rtable(cmp, y - a);
            }
         }
      // swap 'this' and 'prior' lists
      swap(Fthis, Fprev);
      }
   // Compute forward metric for known drift, and return
   real result = 0;
   // event must fit the received sequence:
   // 1. n-1+a >= 0
   // 2. n-1+mu < rx.size() [automatically satisfied by definition of mu]
   // limits on insertions and deletions must be respected:
   // 3. mu-a <= I
   // 4. mu-a >= -1
   // note: muoff and a are offset by xmax
   const int muoff = mu + xmax;
   const int amin = max(max(0, muoff - I), xmax + 1 - n);
   const int amax = min(2 * xmax, muoff + 1);
   // check if the last element is a pure deletion
   int amax_act = amax;
   if (muoff - amax < 0)
      {
      result += Fprev[amax] * Rval();
      amax_act--;
      }
   // elements requiring comparison of tx and rx bits
   for (int a = amin; a <= amax_act; ++a)
      {
      // received subsequence has
      // start:  n-1+a
      // length: mu-a+1
      // therefore last element is: start+length-1 = n+mu-1
      const bool cmp = tx(n - 1) != rx(n + mu - 1);
      result += Fprev[a] * Rtable(cmp, muoff - a);
      }
   // return
   gl_result() = result;
   }

float bsid_receive(const vector<bool>& tx, const vector<bool>& rx,
      const matrix<float>& Rtable, const value<float>& Rval, const int I,
      const int xmax, const int N)
   {
   // declare space for result (initialize to allocate and set memory on device)
   value<float> result;
   result = 0;
   // Compute sizes
   const int n = tx.size();
   const int mu = rx.size() - n;
   assert(n <= N);
   assert(labs(mu) <= xmax);
   // start off GPU threads
   int block = 1;
   int grid = 1;
   size_t shm = 2 * (2 * xmax + 1) * sizeof(float);
   bsid_receive_thread<<<grid, block, shm>>> (result, tx, rx, Rtable, Rval, I, xmax, N);
   return result;
   }

} // end namespace
