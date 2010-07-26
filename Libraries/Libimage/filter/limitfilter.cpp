#include "limitfilter.h"
#include "histogram.h"

namespace libimage {

// initialization

template <class T>
void limitfilter<T>::init(const T lo, const T hi)
   {
   m_lo = lo;
   m_hi = hi;
   }

// filter process loop (only updates output matrix)

template <class T>
void limitfilter<T>::process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const
   {
   const int M = in.size().rows();
   const int N = in.size().cols();

   out.init(M, N);

   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         if (in(i, j) < m_lo)
            out(i, j) = m_lo;
         else if (in(i, j) > m_hi)
            out(i, j) = m_hi;
         else
            out(i, j) = in(i, j);
   }

template <class T>
void limitfilter<T>::process(libbase::matrix<T>& m) const
   {
   const int M = m.size().rows();
   const int N = m.size().cols();

   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         if (m(i, j) < m_lo)
            m(i, j) = m_lo;
         else if (m(i, j) > m_hi)
            m(i, j) = m_hi;
   }

// Explicit Realizations

template class limitfilter<double> ;
template class limitfilter<float> ;
template class limitfilter<int> ;

} // end namespace
