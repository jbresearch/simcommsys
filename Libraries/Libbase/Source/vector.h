#ifndef __vector_h
#define __vector_h

#include "config.h"
#include <stdlib.h>
#include <iostream>

namespace libbase {

/*!
   \brief   Generic Vector.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \note Supports the concept of an empty vector

   \note Multiplication and division perform array operations
   
   \warning Unlike most other classes, this class uses stream I/O as
            serialization for loading and saving; they therefore output
            container size together with container elements.
            The serialize methods input/output only the elements.

   \todo Extract non-root vectors as a derived class

   \todo This class needs to be re-designed in a manner that is consistent with
         convention (esp. Matlab) and that is efficient
*/

template <class T>
class vector;

template <class T>
std::ostream& operator<<(std::ostream& s, const vector<T>& x);
template <class T>
std::istream& operator>>(std::istream& s, vector<T>& x);

template <class T>
class vector {
protected:
   bool  m_root;     //!< True if vector contains its own allocated memory
   int   m_xsize;
   T     *m_data;
protected:
   /*! \name Internal functions */
   //! Verifies that object is in a valid state
   void test_invariant();
   // @}
   /*! \name Memory allocation functions */
   //! Allocates memory for x elements (if necessary) and updates xsize
   void alloc(const int x);
   //! If there is memory allocated, free it
   void free();
   //! Set vector to given size, freeing if and as required
   void setsize(const int x);
   // @}
public:
   //! Default constructor (does not initialise elements)
   explicit vector(const int x=0);
   //! Copy constructor
   vector(const vector<T>& x);
   ~vector() { free(); };

   /*! \name Resizing operations */
   /*! \brief Initialize vector to given size
      This method is guaranteed to leave the vector untouched if the size is
      already good, and only reallocated if necessary. This helps reduce
      redundant free/alloc operations.
   */
   void init(const int x) { setsize(x); };
   //! Initialize vector to the size of given vector
   template <class A> void init(const vector<A>& x) { init(x.size()); };
   // @}
   
   /*! \name Vector copy and value initialisation */
   //! Copy elements from an array.
   vector<T>& assign(const T* x, const int n);
   /*! \brief Copies data from another vector without resizing this one
      If the vectors are not the same size, the first 'n' elements are copied,
      where 'n' is the smaller vector's size. If this vector is larger, the
      remaining elements are left untouched.
   */
   vector<T>& copyfrom(const vector<T>& x);
   //! Copies another vector, resizing this one as necessary
   template <class A> vector<T>& operator=(const vector<A>& x);
   //! Sets all vector elements to the given value
   template <class A> vector<T>& operator=(const A x);
   // @}

   // sub-vector access
   /*! \brief Extract a sub-vector as a reference into this vector
      This allows access to sub-vector data without array copying.
   */
   const vector<T> extract(const int start, const int n) const;

   // index operators (perform boundary checking)
   T& operator()(const int x);
   T operator()(const int x) const;

   // information services
   //! Total number of elements
   int size() const { return m_xsize; };

   /*! \name Serialization and stream input & output */
   void serialize(std::ostream& s, char spacer='\t') const;
   void serialize(std::istream& s);
   friend std::ostream& operator<< <>(std::ostream& s, const vector<T>& x);
   friend std::istream& operator>> <>(std::istream& s, vector<T>& x);
   // @}
   
   // arithmetic operations - unary
   vector<T>& operator+=(const vector<T>& x);
   vector<T>& operator-=(const vector<T>& x);
   vector<T>& operator*=(const vector<T>& x);
   vector<T>& operator/=(const vector<T>& x);
   vector<T>& operator+=(const T x);
   vector<T>& operator-=(const T x);
   vector<T>& operator*=(const T x);
   vector<T>& operator/=(const T x);

   // arithmetic operations - binary
   vector<T> operator+(const vector<T>& x) const;
   vector<T> operator-(const vector<T>& x) const;
   vector<T> operator*(const vector<T>& x) const;
   vector<T> operator/(const vector<T>& x) const;
   vector<T> operator+(const T x) const;
   vector<T> operator-(const T x) const;
   vector<T> operator*(const T x) const;
   vector<T> operator/(const T x) const;

   // boolean operations - unary
   vector<T>& operator!();
   vector<T>& operator&=(const vector<T>& x);
   vector<T>& operator|=(const vector<T>& x);
   vector<T>& operator^=(const vector<T>& x);

   // boolean operations - binary
   vector<T> operator&(const vector<T>& x) const;
   vector<T> operator|(const vector<T>& x) const;
   vector<T> operator^(const vector<T>& x) const;

   //! Apply user-defined operation on all elements
   vector<T>& apply(T f(T));

   /*! \name statistical operations */
   //! Find smallest vector element
   T min() const;
   //! Find largest vector element
   T max() const;
   /*! \brief Find smallest vector element
      \param index returns the index for the smallest value
      \param getfirst flag to return first value found (rather than last)
   */
   T min(int& index, const bool getfirst=true) const;
   /*! \brief Find largest vector element
      \param index returns the index for the largest value
      \param getfirst flag to return first value found (rather than last)
   */
   T max(int& index, const bool getfirst=true) const;
   //! Compute the sum of all vector elements
   T sum() const;
   //! Computes the sum of the squares of all vector elements
   T sumsq() const;
   //! Computes the mathematical mean of vector elements
   T mean() const { return sum()/T(size()); };
   //! Computes the variance of vector elements
   T var() const;
   //! Computes the standard deviation of vector elements
   T sigma() const { return sqrt(var()); };
   // @}
};

// internal functions

template <class T>
inline void vector<T>::test_invariant()
   {
   assert(m_xsize >= 0);
   if(m_xsize == 0)
      assert(m_data == NULL);
   else
      assert(m_data != NULL);
   }

// memory allocation functions

template <class T>
inline void vector<T>::alloc(const int x)
   {
   test_invariant();
   assert(x >= 0);
   assert(m_xsize == 0);
   m_xsize = x;
   m_root = true;
   if(x > 0)
      m_data = new T[x];
   else
      m_data = NULL;
   test_invariant();
   }

template <class T>
inline void vector<T>::free()
   {
   test_invariant();
   if(m_root && m_xsize > 0)
      {
      delete[] m_data;
      m_xsize = 0;
      m_data = NULL;
      }
   test_invariant();
   }

template <class T>
inline void vector<T>::setsize(const int x)
   {
   assert(x >= 0);
   if(x==m_xsize)
      return;
   free();
   alloc(x);
   }

// constructor / destructor functions

template <class T>
inline vector<T>::vector(const int x) :
   m_root(true),
   m_xsize(0),
   m_data(NULL)
   {
   alloc(x);
   }

template <class T>
inline vector<T>::vector(const vector<T>& x) :
   m_root(true),
   m_xsize(0),
   m_data(NULL)
   {
   if(x.m_root)
      {
      alloc(x.m_xsize);
      for(int i=0; i<m_xsize; i++)
         m_data[i] = x.m_data[i];
      }
   else
      {
      m_root = x.m_root;
      m_xsize = x.m_xsize;
      m_data = x.m_data;
      }
   }

// vector copy and value initialisation

template <class T>
inline vector<T>& vector<T>::assign(const T* x, const int n)
   {
   setsize(n);
   for(int i=0; i<n; i++)
      m_data[i] = x[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::copyfrom(const vector<T>& x)
   {
   const int xsize = std::min(m_xsize, x.m_xsize);
   for(int i=0; i<xsize; i++)
      m_data[i] = x.m_data[i];
   return *this;
   }

template <class T>
template <class A>
inline vector<T>& vector<T>::operator=(const vector<A>& x)
   {
   setsize(x.m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] = x.m_data[i];
   return *this;
   }

template <class T>
template <class A>
inline vector<T>& vector<T>::operator=(const A x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = x;
   return *this;
   }

// sub-vector access

template <class T>
inline const vector<T> vector<T>::extract(const int start, const int n) const
   {
   vector<T> r;
   r.m_root = false;
   r.m_xsize = n;
   assert(m_xsize >= start+n);
   r.m_data = &m_data[start];
   return r;
   }

// index operators (perform boundary checking)

template <class T>
inline T& vector<T>::operator()(const int x)
   {
   assert(x>=0 && x<m_xsize);
   return m_data[x];
   }

template <class T>
inline T vector<T>::operator()(const int x) const
   {
   assert(x>=0 && x<m_xsize);
   return m_data[x];
   }

// serialization and stream input & output

template <class T>
inline void vector<T>::serialize(std::ostream& s, char spacer) const
   {
   s << m_data[0];
   for(int i=1; i<m_xsize; i++)
      s << spacer << m_data[i];
   s << '\n';
   }

template <class T>
inline void vector<T>::serialize(std::istream& s)
   {
   for(int i=0; i<m_xsize; i++)
      s >> m_data[i];
   }

template <class T>
inline std::ostream& operator<<(std::ostream& s, const vector<T>& x)
   {
   s << x.m_xsize << "\n";
   x.serialize(s);
   return s;
   }

template <class T>
inline std::istream& operator>>(std::istream& s, vector<T>& x)
   {
   int xsize;
   s >> xsize;
   x.setsize(xsize);
   x.serialize(s);
   return s;
   }

// arithmetic operations - unary

template <class T>
inline vector<T>& vector<T>::operator+=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] += x.m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator-=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] -= x.m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator*=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] *= x.m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator/=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] /= x.m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator+=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] += x;
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator-=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] -= x;
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator*=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] *= x;
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator/=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] /= x;
   return *this;
   }

// arithmetic operations - binary

template <class T>
inline vector<T> vector<T>::operator+(const vector<T>& x) const
   {
   vector<T> r = *this;
   r += x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator-(const vector<T>& x) const
   {
   vector<T> r = *this;
   r -= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator*(const vector<T>& x) const
   {
   vector<T> r = *this;
   r *= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator/(const vector<T>& x) const
   {
   vector<T> r = *this;
   r /= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator+(const T x) const
   {
   vector<T> r = *this;
   r += x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator-(const T x) const
   {
   vector<T> r = *this;
   r -= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator*(const T x) const
   {
   vector<T> r = *this;
   r *= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator/(const T x) const
   {
   vector<T> r = *this;
   r /= x;
   return r;
   }

// boolean operations - unary

template <class T>
inline vector<T>& vector<T>::operator!()
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = !m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator&=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] &= x.m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator|=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] |= x.m_data[i];
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator^=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] ^= x.m_data[i];
   return *this;
   }

// boolean operations - binary

template <class T>
inline vector<T> vector<T>::operator&(const vector<T>& x) const
   {
   vector<T> r = *this;
   r &= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator|(const vector<T>& x) const
   {
   vector<T> r = *this;
   r |= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator^(const vector<T>& x) const
   {
   vector<T> r = *this;
   r ^= x;
   return r;
   }

// user-defined operations

template <class T>
inline vector<T>& vector<T>::apply(T f(T))
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = f(m_data[i]);
   return *this;
   }

// statistical operations

template <class T>
inline T vector<T>::min() const
   {
   assert(m_xsize > 0);
   T result = m_data[0];
   for(int i=1; i<m_xsize; i++)
      if(m_data[i] < result)
         result = m_data[i];
   return result;
   }

template <class T>
inline T vector<T>::max() const
   {
   assert(m_xsize > 0);
   T result = m_data[0];
   for(int i=1; i<m_xsize; i++)
      if(m_data[i] > result)
         result = m_data[i];
   return result;
   }

template <class T>
inline T vector<T>::min(int& index, const bool getfirst) const
   {
   assert(m_xsize > 0);
   T result = m_data[0];
   index = 0;
   for(int i=1; i<m_xsize; i++)
      if(m_data[i] < result)
         {
         result = m_data[i];
         index = i;
         }
      else if(!getfirst && m_data[i] == result)
         index = i;
   return result;
   }

template <class T>
inline T vector<T>::max(int& index, const bool getfirst) const
   {
   assert(m_xsize > 0);
   T result = m_data[0];
   index = 0;
   for(int i=1; i<m_xsize; i++)
      if(m_data[i] > result)
         {
         result = m_data[i];
         index = i;
         }
      else if(!getfirst && m_data[i] == result)
         index = i;
   return result;
   }

template <class T>
inline T vector<T>::sum() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      result += m_data[i];
   return result;
   }

template <class T>
inline T vector<T>::sumsq() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      result += m_data[i] * m_data[i];
   return result;
   }

template <class T>
inline T vector<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

}; // end namespace

#endif
