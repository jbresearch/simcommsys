#include "awfilter.h"

namespace libimage {

const libbase::vcs awfilter_version("Adaptive Wiener Filter module (awfilter)", 1.30);

// initialization

template<class T> void awfilter<T>::init(const int d, const double noise)
   {
   m_d = d;
   m_noise = noise;
   }

// parameter estimation (updates internal statistics)

template<class T> void awfilter<T>::reset()
   {
   rvglobal.reset();
   }

template<class T> void awfilter<T>::update(const libbase::matrix<T>& in)
   {
   const int M = in.xsize();
   const int N = in.ysize();

   for(int i=0; i<M; i++)
      {
      display_progress(i, M);
      for(int j=0; j<N; j++)
         {
         // compute mean and variance of neighbouring pixels
         libbase::rvstatistics rv;
         for(int ii=max(i-m_d,0); ii<=min(i+m_d,M-1); ii++)
            for(int jj=max(j-m_d,0); jj<=min(j+m_d,N-1); jj++)
               rv.insert(in(ii,jj));
         // add to the global sum
         rvglobal.insert(rv.var());
         }
      }
   }

template<class T> void awfilter<T>::estimate()
   {
   m_noise = rvglobal.mean();
   libbase::trace << "Noise threshold = " << m_noise << "\n";
   }

// filter process loop (only updates output matrix)

template<class T> void awfilter<T>::process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const
   {
   const int M = in.xsize();
   const int N = in.ysize();

   out.init(M,N);

   for(int i=0; i<M; i++)
      {
      display_progress(i, M);
      for(int j=0; j<N; j++)
         {
         // compute mean and variance of neighbouring pixels
         libbase::rvstatistics rv;
         for(int ii=max(i-m_d,0); ii<=min(i+m_d,M-1); ii++)
            for(int jj=max(j-m_d,0); jj<=min(j+m_d,N-1); jj++)
               rv.insert(in(ii,jj));
         const double mean = rv.mean();
         const double var = rv.var();
         // compute result
         out(i,j) = T(mean + (max<double>(0,var-m_noise) / max<double>(var,m_noise)) * (in(i,j)-mean));
         }
      }
   }

// Explicit Realizations

template class awfilter<double>;
template class awfilter<int>;

}; // end namespace
