#ifndef __vector_h
#define __vector_h

#include "config.h"
#include "size.h"
#include <stdlib.h>
#include <iostream>
#include <map>

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of memory allocation/deallocation
// 3 - Trace memory allocation/deallocation
// NOTE: since this is a header, this is likely to affect other classes as well;
//       please return this to '1' when committing.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

#if DEBUG>=2
//! Associative array to keep track of memory allocation
extern std::map<void*,int> _vector_heap;
#endif

template <class T>
class vector;

/*!
 * \brief   Size specialization for vector.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <>
class size_type<vector> {
private:
   int n; //!< Length of vector in elements
public:
   /*! \brief Principal Constructor
    */
   explicit size_type(int n = 0)
      {
      this->n = n;
      }
   /*! \brief Conversion to integer
    * Returns the number of elements
    */
   operator int() const
      {
      return n;
      }
   /*! \brief Comparison of two size objects
    * Only true if dimensions are the same
    */
   bool operator==(const size_type<vector>& rhs) const
      {
      return (n == rhs.n);
      }
   /*! \brief Number of elements */
   int length() const
      {
      return n;
      }
   /*! \brief Stream output */
   friend std::ostream& operator<<(std::ostream& sout,
         const size_type<vector> r)
      {
      sout << r.n;
      return sout;
      }
   /*! \brief Stream input */
   friend std::istream& operator>>(std::istream& sin, size_type<vector>& r)
      {
      sin >> r.n;
      return sin;
      }
};

/*!
 * \brief   Generic Vector.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \note Supports the concept of an empty vector
 *
 * \note Multiplication and division perform array operations
 *
 * \warning Unlike most other classes, this class uses stream I/O as
 * serialization for loading and saving; they therefore output
 * container size together with container elements.
 * The serialize methods input/output only the elements.
 *
 * \note For root vectors, copy construction or assignment constitute a deep
 * copy; for non-root vectors the operations are shallow (ie. the reference to
 * the original data set is maintained). The actual allocated memory is only
 * released when the relevant root vector is destroyed. There is always a risk
 * that the root vector is destroyed before non-root references, in which case
 * those references become stale.
 *
 * \todo Fix destruction of root vectors to make sure there are no remaining
 * references to the data set.
 *
 * \todo Extract non-root vectors as a derived class
 *
 * \todo This class needs to be re-designed in a manner that is consistent with
 * convention (esp. Matlab) and that is efficient
 *
 * \todo Extract common implementation of copy assignment operators and
 * copy constructor
 *
 * \todo Merge code for extract() and segment()
 *
 * \todo Add construction from initializer_list when possible
 */

template <class T>
class vector {
protected:
   bool m_root; //!< True if vector contains its own allocated memory
   size_type<libbase::vector> m_size;
   T *m_data;
protected:
   /*! \name Internal functions */
   //! Verifies that object is in a valid state
   void test_invariant() const;
   // @}
   /*! \name Memory allocation functions */
   /*! \brief Allocates memory for x elements (if necessary) and updates xsize
    * \note This is only valid for 'root' vectors.
    */
   void alloc(const int n);
   /*! \brief If there is memory allocated, free it
    * \note This is validly called for non-root and for empty vectors
    */
   void free();
   // @}
public:
   /*! \name Law of the Big Three */
   //! Destructor
   ~vector()
      {
      free();
      }
   //! Copy constructor
   vector(const vector<T>& x);
   //! Copy asignment operator
   vector<T>& operator=(const vector<T>& x);
   // @}

   /*! \name Other Constructors */
   //! Default constructor (does not initialise elements)
   explicit vector(const int n = 0);
   //! On-the-fly conversion of vectors
   template <class A>
   explicit vector(const vector<A>& x);
   // @}

   /*! \name Resizing operations */
   /*! \brief Set vector to given size, freeing if and as required
    * This method is guaranteed to leave the vector untouched if the size is
    * already good, and only reallocated if necessary. This helps reduce
    * redundant free/alloc operations.
    * \note This is only valid for 'root' vectors.
    */
   void init(const int n);
   //! Initialize vector to the given size
   void init(const size_type<libbase::vector>& size)
      {
      init(size.length());
      }
   // @}

   /*! \name Vector copy and value initialisation */
   //! Copy elements from an array.
   vector<T>& assign(const T* x, const int n);
   /*! \brief Copies data from another vector without resizing this one
    * If the vectors are not the same size, the first 'n' elements are copied,
    * where 'n' is the smaller vector's size. If this vector is larger, the
    * remaining elements are left untouched.
    */
   vector<T>& copyfrom(const vector<T>& x);
   //! Copies another vector, resizing this one as necessary
   template <class A>
   vector<T>& operator=(const vector<A>& x);
   /*! \brief Sets all vector elements to the given value
    * There is an advantage in using overloaded '=' instead of an init_value()
    * method: this works even with nested vectors.
    */
   template <class A>
   vector<T>& operator=(const A x);
   // @}

   // sub-vector access
   /*! \brief Extract a sub-vector as a reference into this vector
    * This allows access to sub-vector data without array copying.
    */
   const vector<T> extract(const int start, const int n) const;
   /*! \brief Access part of this vector as a sub-vector
    * This allows operations on sub-vector data without array copying.
    */
   vector<T> segment(const int start, const int n);

   // index operators (perform boundary checking)
   T& operator()(const int x);
   T operator()(const int x) const;

   // information services
   //! Total number of elements
   size_type<libbase::vector> size() const
      {
      return m_size;
      }

   /*! \name Serialization */
   void serialize(std::ostream& sout, char spacer = '\t') const;
   void serialize(std::istream& sin);
   // @}

   /*! \name Direct comparison operations */
   bool isequalto(const vector<T>& x) const;
   bool isnotequalto(const vector<T>& x) const;
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
    * \param index returns the index for the smallest value
    * \param getfirst flag to return first value found (rather than last)
    */
   T min(int& index, const bool getfirst = true) const;
   /*! \brief Find largest vector element
    * \param index returns the index for the largest value
    * \param getfirst flag to return first value found (rather than last)
    */
   T max(int& index, const bool getfirst = true) const;
   //! Compute the sum of all vector elements
   T sum() const;
   //! Computes the sum of the squares of all vector elements
   T sumsq() const;
   //! Computes the mathematical mean of vector elements
   T mean() const
      {
      return sum() / T(size());
      }
   //! Computes the variance of vector elements
   T var() const;
   //! Computes the standard deviation of vector elements
   T sigma() const
      {
      return sqrt(var());
      }
   // @}
};

// internal functions

template <class T>
inline void vector<T>::test_invariant() const
   {
   // size must be valid
   assert(m_size.length() >= 0);
   // pointer must make sense
   if (m_size.length() == 0)
      assert(m_data == NULL);
   else
      assert(m_data != NULL);
#if DEBUG>=2
   if(m_root && m_data != NULL)
      {
      // root vectors: record of allocated memory of correct size
      assert(_vector_heap.count(m_data) > 0);
      assert(_vector_heap[m_data] == m_size.length() * int(sizeof(T)));
      }
#endif
   }

// memory allocation functions

template <class T>
inline void vector<T>::alloc(const int n)
   {
   test_invariant();
   assert(n >= 0);
   assert(m_root);
   assert(m_size.length() == 0);
   m_size = size_type<libbase::vector> (n);
   if (n > 0)
      {
      m_data = new T[n];
#if DEBUG>=2
      assert(_vector_heap.count(m_data) == 0);
      _vector_heap[m_data] = m_size.length() * sizeof(T);
#endif
#if DEBUG>=3
      trace << "DEBUG (vector): allocated " << m_size.length() << " x " << sizeof(T)
      << " bytes at " << m_data << "\n";
#endif
      }
   else
      m_data = NULL;
   test_invariant();
   }

template <class T>
inline void vector<T>::free()
   {
   test_invariant();
   if (m_root && m_size.length() > 0)
      {
#if DEBUG>=2
      assert(_vector_heap.count(m_data) > 0);
      assert(_vector_heap[m_data] == m_size.length() * int(sizeof(T)));
      _vector_heap.erase(_vector_heap.find(m_data));
#endif
#if DEBUG>=3
      trace << "DEBUG (vector): freeing " << m_size.length() << " x " << sizeof(T)
      << " bytes at " << m_data << "\n";
#endif
      delete[] m_data;
      m_size = size_type<libbase::vector> (0);
      m_data = NULL;
      }
   else if (!m_root)
      {
      m_root = true;
      m_size = size_type<libbase::vector> (0);
      m_data = NULL;
      }
   test_invariant();
   }

// constructor / destructor functions

template <class T>
inline vector<T>::vector(const int n) :
   m_root(true), m_size(0), m_data(NULL)
   {
   test_invariant();
   alloc(n);
   test_invariant();
   }

template <class T>
inline vector<T>::vector(const vector<T>& x) :
   m_root(true), m_size(0), m_data(NULL)
   {
   test_invariant();
   if (x.m_root)
      {
      alloc(x.m_size.length());
      for (int i = 0; i < m_size.length(); i++)
         m_data[i] = x.m_data[i];
      }
   else
      {
      m_root = x.m_root;
      m_size = x.m_size;
      m_data = x.m_data;
      }
   test_invariant();
   }

template <class T>
template <class A>
inline vector<T>::vector(const vector<A>& x) :
   m_root(true), m_size(0), m_data(NULL)
   {
   test_invariant();
   init(x.size());
   // avoid down-cast warnings in Win32
#ifdef WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 4800 )
#endif
   // Do not convert type of element from A to T, so that if either is a
   // vector, the process can continue through the assignment operator
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x(i);
#ifdef WIN32
#  pragma warning( pop )
#endif
   test_invariant();
   }

// Resizing operations

template <class T>
inline void vector<T>::init(const int n)
   {
   test_invariant();
   assert(n >= 0);
   if (n == m_size.length())
      return;
   assert(m_root);
   free();
   alloc(n);
   test_invariant();
   }

// vector copy and value initialisation

template <class T>
inline vector<T>& vector<T>::assign(const T* x, const int n)
   {
   test_invariant();
   init(n);
   for (int i = 0; i < n; i++)
      m_data[i] = x[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::copyfrom(const vector<T>& x)
   {
   test_invariant();
   const int xsize = std::min(m_size.length(), x.m_size.length());
   for (int i = 0; i < xsize; i++)
      m_data[i] = x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator=(const vector<T>& x)
   {
   test_invariant();
   // correctly handle self-assignment
   if (this == &x)
      return *this;
   init(x.size());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x(i);
   test_invariant();
   return *this;
   }

template <class T>
template <class A>
inline vector<T>& vector<T>::operator=(const vector<A>& x)
   {
   test_invariant();
   // this should never correspond to self-assignment
   assert((void *)this != (void *)&x);
   init(x.size());
   // avoid down-cast warnings in Win32
#ifdef WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 )
#endif
   // Do not convert type of element from A to T, so that if either is a
   // vector, the process can continue recursively
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x(i);
#ifdef WIN32
#  pragma warning( pop )
#endif
   test_invariant();
   return *this;
   }

template <class T>
template <class A>
inline vector<T>& vector<T>::operator=(const A x)
   {
   test_invariant();
   // avoid down-cast warnings in Win32
#ifdef WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 )
#endif
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x;
#ifdef WIN32
#  pragma warning( pop )
#endif
   test_invariant();
   return *this;
   }

// sub-vector access

template <class T>
inline const vector<T> vector<T>::extract(const int start, const int n) const
   {
   test_invariant();
   assert(n >= 0);
   vector<T> r;
   r.m_root = false;
   r.m_size = size_type<libbase::vector> (n);
   assert(m_size.length() >= start+n);
   r.m_data = (n > 0) ? &m_data[start] : NULL;
   r.test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::segment(const int start, const int n)
   {
   test_invariant();
   assert(n >= 0);
   vector<T> r;
   r.m_root = false;
   r.m_size = size_type<libbase::vector> (n);
   assert(m_size.length() >= start+n);
   r.m_data = (n > 0) ? &m_data[start] : NULL;
   r.test_invariant();
   return r;
   }

// index operators (perform boundary checking)

template <class T>
inline T& vector<T>::operator()(const int x)
   {
   test_invariant();
   assert(x>=0 && x<m_size.length());
   return m_data[x];
   }

template <class T>
inline T vector<T>::operator()(const int x) const
   {
   test_invariant();
   assert(x>=0 && x<m_size.length());
   return m_data[x];
   }

// serialization and stream input & output

template <class T>
inline void vector<T>::serialize(std::ostream& sout, char spacer) const
   {
   test_invariant();
   if (m_size.length() > 0)
      sout << m_data[0];
   for (int i = 1; i < m_size.length(); i++)
      sout << spacer << m_data[i];
   sout << '\n';
   }

template <class T>
inline void vector<T>::serialize(std::istream& sin)
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      sin >> m_data[i];
   test_invariant();
   }

template <class T>
inline std::ostream& operator<<(std::ostream& s, const vector<T>& x)
   {
   s << x.size() << "\n";
   x.serialize(s);
   return s;
   }

template <class T>
inline std::istream& operator>>(std::istream& s, vector<T>& x)
   {
   size_type<vector> size;
   s >> size;
   x.init(size);
   x.serialize(s);
   return s;
   }

// direct comparison operations

template <class T>
inline bool vector<T>::isequalto(const vector<T>& x) const
   {
   if (x.m_size != m_size)
      return false;
   for (int i = 0; i < m_size.length(); i++)
      if (m_data[i] != x.m_data[i])
         return false;
   return true;
   }

template <class T>
inline bool vector<T>::isnotequalto(const vector<T>& x) const
   {
   return !isequalto(x);
   }

// arithmetic operations - unary

template <class T>
inline vector<T>& vector<T>::operator+=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] += x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator-=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] -= x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator*=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] *= x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator/=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] /= x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator+=(const T x)
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] += x;
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator-=(const T x)
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] -= x;
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator*=(const T x)
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] *= x;
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator/=(const T x)
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] /= x;
   test_invariant();
   return *this;
   }

// arithmetic operations - binary

template <class T>
inline vector<T> vector<T>::operator+(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r += x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator-(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r -= x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator*(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r *= x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator/(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r /= x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator+(const T x) const
   {
   test_invariant();
   vector<T> r = *this;
   r += x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator-(const T x) const
   {
   test_invariant();
   vector<T> r = *this;
   r -= x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator*(const T x) const
   {
   test_invariant();
   vector<T> r = *this;
   r *= x;
   test_invariant();
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator/(const T x) const
   {
   test_invariant();
   vector<T> r = *this;
   r /= x;
   test_invariant();
   return r;
   }

// boolean operations - unary

template <class T>
inline vector<T>& vector<T>::operator!()
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = !m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator&=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] &= x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator|=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] |= x.m_data[i];
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::operator^=(const vector<T>& x)
   {
   test_invariant();
   assert(x.m_size.length() == m_size.length());
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] ^= x.m_data[i];
   test_invariant();
   return *this;
   }

// boolean operations - binary

template <class T>
inline vector<T> vector<T>::operator&(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r &= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator|(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r |= x;
   return r;
   }

template <class T>
inline vector<T> vector<T>::operator^(const vector<T>& x) const
   {
   test_invariant();
   vector<T> r = *this;
   r ^= x;
   return r;
   }

// user-defined operations

template <class T>
inline vector<T>& vector<T>::apply(T f(T))
   {
   test_invariant();
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = f(m_data[i]);
   test_invariant();
   return *this;
   }

// statistical operations

template <class T>
inline T vector<T>::min() const
   {
   test_invariant();
   assertalways(m_size.length() > 0);
   T result = m_data[0];
   for (int i = 1; i < m_size.length(); i++)
      if (m_data[i] < result)
         result = m_data[i];
   test_invariant();
   return result;
   }

template <class T>
inline T vector<T>::max() const
   {
   test_invariant();
   assertalways(m_size.length() > 0);
   T result = m_data[0];
   for (int i = 1; i < m_size.length(); i++)
      if (m_data[i] > result)
         result = m_data[i];
   test_invariant();
   return result;
   }

template <class T>
inline T vector<T>::min(int& index, const bool getfirst) const
   {
   test_invariant();
   assertalways(m_size.length() > 0);
   T result = m_data[0];
   index = 0;
   for (int i = 1; i < m_size.length(); i++)
      if (m_data[i] < result)
         {
         result = m_data[i];
         index = i;
         }
      else if (!getfirst && m_data[i] == result)
         index = i;
   test_invariant();
   return result;
   }

template <class T>
inline T vector<T>::max(int& index, const bool getfirst) const
   {
   test_invariant();
   assertalways(m_size.length() > 0);
   T result = m_data[0];
   index = 0;
   for (int i = 1; i < m_size.length(); i++)
      if (m_data[i] > result)
         {
         result = m_data[i];
         index = i;
         }
      else if (!getfirst && m_data[i] == result)
         index = i;
   test_invariant();
   return result;
   }

template <class T>
inline T vector<T>::sum() const
   {
   test_invariant();
   assertalways(m_size.length() > 0);
   T result = 0;
   for (int i = 0; i < m_size.length(); i++)
      result += m_data[i];
   test_invariant();
   return result;
   }

template <class T>
inline T vector<T>::sumsq() const
   {
   test_invariant();
   assertalways(m_size.length() > 0);
   T result = 0;
   for (int i = 0; i < m_size.length(); i++)
      result += m_data[i] * m_data[i];
   test_invariant();
   return result;
   }

template <class T>
inline T vector<T>::var() const
   {
   test_invariant();
   const T _mean = mean();
   const T _var = sumsq() / T(size()) - _mean * _mean;
   test_invariant();
   return (_var > 0) ? _var : 0;
   }

} // end namespace

#endif
