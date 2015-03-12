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

#ifndef __vector_h
#define __vector_h

#include "config.h"
#include "size.h"
#include "aligned_allocator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <typeinfo>

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of memory allocation/deallocation
// 3 - Trace memory allocation/deallocation
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

#if DEBUG>=2
//! Associative array to keep track of memory allocation of main arrays
extern std::map<const void*, int> _vector_heap;
//! Associative array to keep track of references to main arrays
extern std::map<std::pair<const void*, int>, int> _vector_refs;
#endif

template <class T>
class vector;
template <class T>
class indirect_vector;
template <class T>
class masked_vector;

/*!
 * \brief   Size specialization for vector.
 * \author  Johann Briffa
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
 * \note Supports the concept of an empty vector
 *
 * \note Multiplication and division perform array operations
 *
 * \warning Unlike most other classes, this class uses stream I/O as
 * serialization for loading and saving; they therefore output
 * container size together with container elements.
 * The serialize methods input/output only the elements.
 *
 *
 * \todo Use vector-processing where available
 *
 * \todo Merge code for extract() and segment()
 *
 * \todo Add construction from initializer_list when possible
 */

template <class T>
class vector {
   friend class indirect_vector<T> ;
   friend class masked_vector<T> ;
protected:
   typedef std::allocator<T> Allocator;
   Allocator allocator;
   size_type<libbase::vector> m_size;
   T *m_data;
protected:
   /*! \name Internal functions */
   //! Verifies that object is in a valid state
   void test_invariant() const;
   //! Records m_data as a newly allocated array
   void record_allocation() const;
   //! Destroys record of m_data allocation, prior to freeing
   void remove_allocation() const;
   //! Validates the current pointer from records
   void validate_allocation() const;
   // @}
   /*! \name Memory allocation functions */
   /*! \brief Allocates memory and updates internal size
    * \note This can only be called when the vector is empty.
    */
   void alloc(const int n);
   /*! \brief Deallocates memory only (does not update internal size or pointer)
    * \note This can only be called when the vector is not empty.
    */
   void dealloc();
   /*! \brief If there is memory allocated, free it
    * \note This is validly called for empty vectors, in which case it does
    * nothing.
    */
   void free();
   /*! \brief Copy 'n' elements from source to destination
    * \note Arrays must be non-overlapping, and self-copy is not allowed
    */
   void copy(T* dst, const T* src, int n);
   // @}
public:
   /*! \name Law of the Big Three */
   //! Destructor
   virtual ~vector()
      {
      if (m_size.length() > 0)
         dealloc();
      }
   /*! \brief Copy constructor
    * \note Copy construction is a deep copy.
    */
   vector(const vector<T>& x);
   /*! \brief Copy assignment operator
    * \note Copy assignment is a deep copy.
    */
   vector<T>& operator=(const vector<T>& x);
   // @}

   /*! \name Other Constructors */
   /*! \brief Default constructor
    * Allocates space as requested, but does not initialize elements.
    */
   explicit vector(const int n = 0);
   /*! \brief On-the-fly conversion of vectors
    * \note Naturally this requires a deep copy.
    */
   template <class A>
   explicit vector(const vector<A>& x);
   /*! \brief On-the-fly conversion from STL vector
    */
   template <class A>
   explicit vector(const std::vector<A>& x);
   // @}

   /*! \name Vector copy and value initialisation */
   /*! \brief Copy array.
    * Assigns vector values from array, resizing as necessary.
    */
   vector<T>& assign(const T* x, const int n);
   /*! \brief Copies data from another vector without resizing this one
    * If the vectors are not the same size, the first 'n' elements are copied,
    * where 'n' is the smaller vector's size. If this vector is larger, the
    * remaining elements are left untouched.
    */
   vector<T>& copyfrom(const vector<T>& x);
   /*! \brief Auto-converting copy assignment
    * \note Naturally this requires a deep copy.
    */
   template <class A>
   vector<T>& operator=(const vector<A>& x);
   /*! \brief Sets all vector elements to the given value
    * \note There is an advantage in using overloaded '=' instead of an
    * init_value() method: this works even with nested vectors.
    */
   template <class A>
   vector<T>& operator=(const A x);
   /*! \brief Auto-converting copy assignment for indirect vectors
    * \note Naturally this requires a deep copy.
    */
   template <class A>
   vector<T>& operator=(const indirect_vector<A>& x)
      {
      *this = dynamic_cast<const vector<A>&> (x);
      return *this;
      }
   // @}

   /*! \brief Swap two vectors (constant time)
    *  This exchanges the elements between two vectors in constant time.
    *  This is meant to be picked up instead of the global std::swap() function
    *  using argument-dependent lookup.
    *  \todo Test this before inclusion.
    */
   /*
   friend void swap(vector<T>& lhs, vector<T>& rhs)
      {
      std::swap(lhs.allocator, rhs.allocator);
      std::swap(lhs.m_data, rhs.m_data);
      std::swap(lhs.m_size, rhs.m_size);
      }
      */

   /*! \name Resizing operations */
   /*! \brief Set vector to given size, freeing if and as required
    * This method is guaranteed to leave the vector untouched if the size is
    * already good, and only reallocated if necessary. This helps reduce
    * redundant free/alloc operations.
    */
   void init(const int n);
   //! Initialize vector to the given size
   void init(const size_type<libbase::vector>& size)
      {
      init(size.length());
      }
   // @}

   /*! \name Element access */
   /*! \brief Extract a sub-vector as a reference into this vector
    * This allows read access to sub-vector data without array copying.
    */
   const indirect_vector<T> extract(const int start, const int n) const
      {
      return indirect_vector<T> (const_cast<vector<T>&> (*this), start, n);
      }
   /*! \brief Access part of this vector as a sub-vector
    * This allows write access to sub-vector data without array copying.
    */
   indirect_vector<T> segment(const int start, const int n)
      {
      return indirect_vector<T> (*this, start, n);
      }
   /*! \brief Bind a mask to a vector
    */
   masked_vector<T> mask(const vector<bool>& x)
      {
      return masked_vector<T> (this, x);
      }
   /*! \brief Bind a mask to a vector (read only)
    */
   const masked_vector<T> mask(const vector<bool>& x) const
      {
      return masked_vector<T> (const_cast<vector<T>*>(this), x);
      }
   /*! \brief Index operator (write-access)
    * \note Performs boundary checking.
    */
   T& operator()(const int x)
      {
      test_invariant();
      assert(x >= 0 && x < m_size.length());
      return m_data[x];
      }
   /*! \brief Index operator (read-only access)
    * \note Performs boundary checking.
    */
   const T& operator()(const int x) const
      {
      test_invariant();
      assert(x >= 0 && x < m_size.length());
      return m_data[x];
      }
   // @}

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

   /*! \name Comparison (mask-creation) operations */
   vector<bool> operator==(const vector<T>& x) const;
   vector<bool> operator!=(const vector<T>& x) const;
   vector<bool> operator<=(const vector<T>& x) const;
   vector<bool> operator>=(const vector<T>& x) const;
   vector<bool> operator<(const vector<T>& x) const;
   vector<bool> operator>(const vector<T>& x) const;
   vector<bool> operator==(const T x) const;
   vector<bool> operator!=(const T x) const;
   vector<bool> operator<=(const T x) const;
   vector<bool> operator>=(const T x) const;
   vector<bool> operator<(const T x) const;
   vector<bool> operator>(const T x) const;
   // @}

   /*! \name Direct comparison operations */
   bool isequalto(const vector<T>& x) const;
   bool isnotequalto(const vector<T>& x) const;
   // @}

   /*! \name Arithmetic operations - unary */
   vector<T>& operator+=(const vector<T>& x);
   vector<T>& operator-=(const vector<T>& x);
   vector<T>& operator*=(const vector<T>& x);
   vector<T>& operator/=(const vector<T>& x);
   vector<T>& operator+=(const T x);
   vector<T>& operator-=(const T x);
   vector<T>& operator*=(const T x);
   vector<T>& operator/=(const T x);
   // @}

   /*! \name Arithmetic operations - binary */
   vector<T> operator+(const vector<T>& x) const;
   vector<T> operator-(const vector<T>& x) const;
   vector<T> operator*(const vector<T>& x) const;
   vector<T> operator/(const vector<T>& x) const;
   vector<T> operator+(const T x) const;
   vector<T> operator-(const T x) const;
   vector<T> operator*(const T x) const;
   vector<T> operator/(const T x) const;
   // @}

   /*! \name Boolean operations - modifying */
   vector<T>& operator&=(const vector<T>& x);
   vector<T>& operator|=(const vector<T>& x);
   vector<T>& operator^=(const vector<T>& x);
   // @}

   /*! \name Boolean operations - non-modifying */
   vector<T> operator!();
   vector<T> operator&(const vector<T>& x) const;
   vector<T> operator|(const vector<T>& x) const;
   vector<T> operator^(const vector<T>& x) const;
   // @}

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
   // check records for existing allocated memory of correct size
   //validate_allocation();
   }

template <class T>
inline void vector<T>::record_allocation() const
   {
#if DEBUG>=3
   trace << "DEBUG (vector): allocated " << m_size.length() << " x "
   << sizeof(T) << " bytes at " << m_data << std::endl;
#endif
#if DEBUG>=2
   // confirm there was no prior allocation at this space
   assert(m_data != NULL);
   assert(_vector_heap.count(m_data) == 0);
   // record this allocation
   _vector_heap[m_data] = m_size.length() * sizeof(T);
#endif
   }

template <class T>
inline void vector<T>::remove_allocation() const
   {
#if DEBUG>=3
   trace << "DEBUG (vector): freeing " << m_size.length() << " x " << sizeof(T)
   << " bytes at " << m_data << std::endl;
#endif
#if DEBUG>=2
   // first confirm any existing allocation is properly recorded
   validate_allocation();
   // confirm there are no dangling references
   assert(m_data != NULL);
   std::pair<const void*, int> ndx(m_data, m_size.length() * sizeof(T));
   assert(_vector_refs.count(ndx) == 0);
   // remove record of this allocation
   _vector_heap.erase(_vector_heap.find(m_data));
#endif
   }

template <class T>
inline void vector<T>::validate_allocation() const
   {
#if DEBUG>=2
   if (m_data != NULL)
      {
      // confirm a record exists of existing allocation with the correct value
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
   assert(m_size.length() == 0);
   m_size = size_type<libbase::vector> (n);
   if (n > 0)
      {
      // allocate memory for all elements
      m_data = allocator.allocate(n);
      record_allocation();
      // call default constructor
      const T element = T();
      for (int i = 0; i < n; i++)
         allocator.construct(&m_data[i], element);
      }
   else
      m_data = NULL;
   test_invariant();
   }

template <class T>
inline void vector<T>::dealloc()
   {
   test_invariant();
   assert(m_size.length() > 0);
   remove_allocation();
   // call destructor
   const int n = m_size.length();
   for (int i = 0; i < n; i++)
      allocator.destroy(&m_data[i]);
   // deallocate memory
   allocator.deallocate(m_data, n);
   }

template <class T>
inline void vector<T>::free()
   {
   test_invariant();
   if (m_size.length() > 0)
      {
      dealloc();
      // reset fields
      m_size = size_type<libbase::vector> (0);
      m_data = NULL;
      }
   test_invariant();
   }

template <class T>
inline void vector<T>::copy(T* dst, const T* src, int n)
   {
   test_invariant();
   // handle empty copy correctly
   if (n == 0)
      return;
   // check for self-copy
   assert(dst != src);
   // determine the required amount of data to copy
   const int nbytes = n * sizeof(T);
#ifndef NDEBUG
   // check for non-overlapping arrays
   const int8u* buf1 = std::min((int8u*) dst, (int8u*) src);
   const int8u* buf2 = std::max((int8u*) dst, (int8u*) src);
   assert(buf2 >= buf1 + nbytes);
#endif
   // do the copy
   // NOTE: we can only use memory-copy if T is a primitive type
   if (typeid(T) == typeid(bool) || typeid(T) == typeid(int) || typeid(T)
         == typeid(double) || typeid(T) == typeid(float))
      memcpy(dst, src, nbytes);
   else
      for (int i = 0; i < n; i++)
         dst[i] = src[i];
   test_invariant();
   }

// constructor / destructor functions

template <class T>
inline vector<T>::vector(const int n) :
   m_size(0), m_data(NULL)
   {
   test_invariant();
   alloc(n);
   test_invariant();
   }

template <class T>
inline vector<T>::vector(const vector<T>& x) :
   m_size(0), m_data(NULL)
   {
   test_invariant();
   alloc(x.m_size.length());
   copy(m_data, x.m_data, m_size.length());
   test_invariant();
   }

template <class T>
template <class A>
inline vector<T>::vector(const vector<A>& x) :
   m_size(0), m_data(NULL)
   {
   test_invariant();
   alloc(x.size().length());
   // avoid down-cast warnings in Win32
#ifdef _WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 4800 )
#endif
   // Do not convert type of element from A to T, so that if either is a
   // vector, the process can continue through the assignment operator
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x(i);
#ifdef _WIN32
#  pragma warning( pop )
#endif
   test_invariant();
   }

template <class T>
template <class A>
inline vector<T>::vector(const std::vector<A>& x) :
   m_size(0), m_data(NULL)
   {
   test_invariant();
   alloc(x.size());
   // avoid down-cast warnings in Win32
#ifdef _WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 4800 )
#endif
   // Do not convert type of element from A to T, so that if either is a
   // vector, the process can continue through the assignment operator
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x[i];
#ifdef _WIN32
#  pragma warning( pop )
#endif
   test_invariant();
   }

// vector copy and value initialisation

template <class T>
inline vector<T>& vector<T>::assign(const T* x, const int n)
   {
   test_invariant();
   init(n);
   copy(m_data, x, n);
   test_invariant();
   return *this;
   }

template <class T>
inline vector<T>& vector<T>::copyfrom(const vector<T>& x)
   {
   test_invariant();
   const int xsize = std::min(m_size.length(), x.m_size.length());
   copy(m_data, x.m_data, xsize);
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
   copy(m_data, x.m_data, m_size.length());
   test_invariant();
   return *this;
   }

template <class T>
template <class A>
inline vector<T>& vector<T>::operator=(const vector<A>& x)
   {
   test_invariant();
   // this should never correspond to self-assignment
   assert((void *) this != (void *) &x);
   init(x.size());
   // avoid down-cast warnings in Win32
#ifdef _WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 )
#endif
   // Do not convert type of element from A to T, so that if either is a
   // vector, the process can continue recursively
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x(i);
#ifdef _WIN32
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
#ifdef _WIN32
#  pragma warning( push )
#  pragma warning( disable : 4244 )
#  pragma warning( disable : 4800 )
#endif
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = x;
#ifdef _WIN32
#  pragma warning( pop )
#endif
   test_invariant();
   return *this;
   }

// Resizing operations

template <class T>
inline void vector<T>::init(const int n)
   {
   test_invariant();
   assert(n >= 0);
   // short-cut if correct size is already allocated
   if (n == m_size.length())
      return;
   // otherwise free and reallocate
   free();
   alloc(n);
   test_invariant();
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
   sout << std::endl;
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
   s << x.size() << std::endl;
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

// comparison (mask-creation) operations

template <class T>
inline vector<bool> vector<T>::operator==(const vector<T>& x) const
   {
   assert(x.m_size == m_size);
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] == x.m_data[i]);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator!=(const vector<T>& x) const
   {
   assert(x.m_size == m_size);
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] != x.m_data[i]);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator<=(const vector<T>& x) const
   {
   assert(x.m_size == m_size);
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] <= x.m_data[i]);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator>=(const vector<T>& x) const
   {
   assert(x.m_size == m_size);
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] >= x.m_data[i]);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator<(const vector<T>& x) const
   {
   assert(x.m_size == m_size);
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] < x.m_data[i]);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator>(const vector<T>& x) const
   {
   assert(x.m_size == m_size);
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] > x.m_data[i]);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator==(const T x) const
   {
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] == x);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator!=(const T x) const
   {
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] != x);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator<=(const T x) const
   {
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] <= x);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator>=(const T x) const
   {
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] >= x);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator<(const T x) const
   {
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] < x);
   return r;
   }

template <class T>
inline vector<bool> vector<T>::operator>(const T x) const
   {
   vector<bool> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r(i) = (m_data[i] > x);
   return r;
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
   // avoid bool-related warnings in Win32
#ifdef _WIN32
#  pragma warning( push )
#  pragma warning( disable : 4804 4800 )
#endif
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] += x.m_data[i];
#ifdef _WIN32
#  pragma warning( pop )
#endif
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

// boolean operations - modifying

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

// boolean operations - non-modifying

template <class T>
inline vector<T> vector<T>::operator!()
   {
   test_invariant();
   vector<T> r(m_size);
   for (int i = 0; i < m_size.length(); i++)
      r.m_data[i] = !m_data[i];
   test_invariant();
   return r;
   }

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

/*!
 * \brief   Indirect Vector.
 * \author  Johann Briffa
 *
 * An indirect vector is a vector that does not own its allocated memory.
 * Consequently, all operations that require a resize are forbidden.
 * The data set is really just a reference to (part of) a regular vector.
 * When an indirect vector is destroyed, the actual allocated memory is not
 * released. This only happens when the referenced vector is destroyed.
 * There is always a risk that the referenced vector is destroyed before
 * the indirect references, in which case those references become stale.
 *
 * It is intended that for the user, the use of indirect vectors should be
 * essentially transparent (in that they can mostly be used in place of
 * normal vectors) and that the user should never create one explicitly,
 * but merely through a normal vector.
 */
template <class T>
class indirect_vector : public vector<T> {
   friend class vector<T> ;
   typedef vector<T> Base;
protected:
#if DEBUG>=2
   const size_type<libbase::vector> r_size;
   const T *r_data;
#endif
protected:
   /*! \name Internal functions */
   //! Records m_data as a newly allocated array
   void record_reference() const;
   //! Destroys record of m_data allocation, prior to freeing
   void remove_reference() const;
   //! Validates the current pointer from records
   void validate_reference() const;
   // @}
   /*! \name Constructors */
   //! Unique constructor, can be called only by friends
   indirect_vector(vector<T>& x, const int start, const int n)
#if DEBUG>=2
   :
   r_size(x.size()), r_data(x.m_data)
#endif
      {
      Base::test_invariant();
      assert(start >= 0);
      assert(n >= 0);
      assert(start + n <= x.size().length());
      // update base class by shallow copy, if necessary
      if (n > 0)
         {
         Base::m_size = size_type<libbase::vector> (n);
         Base::m_data = &x(start);
         }
      record_reference();
      Base::test_invariant();
      }
   // @}
public:
   /*! \name Constructors */
   //! Unique constructor
   indirect_vector(T* start, const int n)
#if DEBUG>=2
   :
   r_size(n), r_data(start)
#endif
      {
      Base::test_invariant();
      assert(start);
      assert(n >= 0);
      // update base class by shallow copy, if necessary
      if (n > 0)
         {
         Base::m_size = size_type<libbase::vector> (n);
         Base::m_data = start;
         }
      record_reference();
      Base::test_invariant();
      }
   // @}
   /*! \name Law of the Big Three */
   //! Destructor
   virtual ~indirect_vector()
      {
      // reset base class, in preparation for eventual destruction
      Base::m_size = size_type<libbase::vector> (0);
      Base::m_data = NULL;
      remove_reference();
      }
   /*! \brief Copy constructor
    * \note Copy construction is a shallow copy.
    */
   indirect_vector(const indirect_vector<T>& x)
#if DEBUG>=2
   :
   r_size(x.r_size), r_data(x.r_data)
#endif
      {
      Base::m_size = x.m_size;
      Base::m_data = x.m_data;
      record_reference();
      }
   /*! \brief Copy assignment operator
    * \note Copy assignment is a deep copy.
    * \note This operation is only defined if the size is already correct.
    */
   indirect_vector<T>& operator=(const indirect_vector<T>& x)
      {
      assert(Base::m_size == x.m_size);
      this->copy(Base::m_data, x.m_data, Base::m_size.length());
      return *this;
      }
   // @}

   /*! \brief Sets all vector elements to the given value
    * \note There is an advantage in using overloaded '=' instead of an
    * init_value() method: this works even with nested vectors.
    */
   template <class A>
   indirect_vector<T>& operator=(const A x)
      {
      Base::operator=(x);
      return *this;
      }

   /*! \brief Auto-converting copy assignment for vectors
    * \note Naturally this requires a deep copy.
    * \note This operation is only defined if the size is already correct.
    */
   template <class A>
   indirect_vector<T>& operator=(const vector<A>& x)
      {
      assert(Base::m_size == x.size());
      dynamic_cast<vector<T>&> (*this) = x;
      return *this;
      }
};

// internal functions

template <class T>
inline void indirect_vector<T>::record_reference() const
   {
#if DEBUG>=3
   trace << "DEBUG (indirect_vector): referenced " << r_size.length() << " x "
   << sizeof(T) << " bytes at " << r_data << std::endl;
#endif
#if DEBUG>=2
   assert(r_data != NULL);
   std::pair<const void*, int> ndx(r_data, r_size.length() * sizeof(T));
   // create record if this is a new ref, otherwise increase ref count
   if (_vector_refs.count(ndx) == 0)
   _vector_refs[ndx] = 1;
   else
   _vector_refs[ndx]++;
#endif
   }

template <class T>
inline void indirect_vector<T>::remove_reference() const
   {
#if DEBUG>=3
   trace << "DEBUG (indirect_vector): removing ref to " << r_size.length()
   << " x " << sizeof(T) << " bytes at " << r_data << std::endl;
#endif
#if DEBUG>=2
   // first confirm a reference to this data set is recorded
   validate_reference();
   std::pair<const void*, int> ndx(r_data, r_size.length() * sizeof(T));
   // reduce count, and remove if no more references
   _vector_refs[ndx]--;
   if (_vector_refs[ndx] == 0)
   _vector_refs.erase(_vector_refs.find(ndx));
#endif
   }

template <class T>
inline void indirect_vector<T>::validate_reference() const
   {
#if DEBUG>=2
   assert(r_data != NULL);
   std::pair<const void*, int> ndx(r_data, r_size.length() * sizeof(T));
   assert(_vector_refs.count(ndx) > 0);
   assert(_vector_refs[ndx] > 0);
#endif
   }

/*!
 * \brief   Masked Vector.
 * \author  Johann Briffa
 *
 * A masked vector is a vector with a binary element-mask. Arithmetic,
 * statistical, user-defined operation, and copy/value init functions are
 * defined for this class, allowing us to modify the masked parts of any
 * given vector with ease.
 *
 * It is intended that for the user, the use of masked vectors should be
 * essentially transparent (in that they can mostly be used in place of
 * normal vectors) and that the user should never create one explicitly,
 * but merely through a normal vector.
 */
template <class T>
class masked_vector {
   friend class vector<T> ;
private:
   vector<T>* m_data;
   vector<bool> m_mask;
protected:
   masked_vector(vector<T>* data, const vector<bool>& mask);
public:
   // vector copy and value initialisation
   masked_vector<T>& operator=(const T x);

   // convert to a vector
   operator vector<T>() const;

   // arithmetic operations - unary
   masked_vector<T>& operator+=(const vector<T>& x);
   masked_vector<T>& operator-=(const vector<T>& x);
   masked_vector<T>& operator*=(const vector<T>& x);
   masked_vector<T>& operator/=(const vector<T>& x);
   masked_vector<T>& operator+=(const T x);
   masked_vector<T>& operator-=(const T x);
   masked_vector<T>& operator*=(const T x);
   masked_vector<T>& operator/=(const T x);

   // user-defined operations
   masked_vector<T>& apply(T f(T));

   // information services
   int size() const; // number of masked-in elements

   // statistical operations
   T min() const;
   T max() const;
   T sum() const;
   T sumsq() const;
   T mean() const
      {
      return sum() / T(size());
      }
   T var() const;
   T sigma() const
      {
      return sqrt(var());
      }
};

// masked vector constructor

template <class T>
inline masked_vector<T>::masked_vector(vector<T>* data,
      const vector<bool>& mask)
      : m_data(data), m_mask(mask)
   {
   assert(data->m_size == mask.size());
   }

// vector copy and value initialisation

template <class T>
inline masked_vector<T>& masked_vector<T>::operator=(const T x)
   {
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] = x;
   return *this;
   }

// convert to a vector

template <class T>
inline masked_vector<T>::operator vector<T>() const
   {
   vector<T> v(size());
   int k = 0;
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         v(k++) = m_data->m_data[i];
   return v;
   }

// arithmetic operations - unary

template <class T>
inline masked_vector<T>& masked_vector<T>::operator+=(const vector<T>& x)
   {
   assert(x.m_size == m_data->m_size);
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] += x.m_data[i];
   return *this;
   }

template <class T>
inline masked_vector<T>& masked_vector<T>::operator-=(const vector<T>& x)
   {
   assert(x.m_size == m_data->m_size);
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] -= x.m_data[i];
   return *this;
   }

/*!
 * \brief Array multiplication (element-by-element) of vectors
 * \param  x   Vector to be multiplied to this one
 * \return The updated (multiplied-into) vector
 *
 * Masked elements (ie. where the mask is true) are multiplied by
 * the corresponding element in 'x'. Unmasked elements are left
 * untouched.
 *
 * \note For A.*B, the size of A must be the same as the size of B.
 */
template <class T>
inline masked_vector<T>& masked_vector<T>::operator*=(const vector<T>& x)
   {
   assert(x.m_size == m_data->m_size);
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] *= x.m_data[i];
   return *this;
   }

/*!
 * \brief Array division (element-by-element) of vectors
 * \param  x   Vector to divide this one by
 * \return The updated (divided-into) vector
 *
 * Masked elements (ie. where the mask is true) are divided by
 * the corresponding element in 'x'. Unmasked elements are left
 * untouched.
 *
 * \note For A./B, the size of A must be the same as the size of B.
 */
template <class T>
inline masked_vector<T>& masked_vector<T>::operator/=(const vector<T>& x)
   {
   assert(x.m_size == m_data->m_size);
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] /= x.m_data[i];
   return *this;
   }

template <class T>
inline masked_vector<T>& masked_vector<T>::operator+=(const T x)
   {
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] += x;
   return *this;
   }

template <class T>
inline masked_vector<T>& masked_vector<T>::operator-=(const T x)
   {
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] -= x;
   return *this;
   }

template <class T>
inline masked_vector<T>& masked_vector<T>::operator*=(const T x)
   {
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] *= x;
   return *this;
   }

template <class T>
inline masked_vector<T>& masked_vector<T>::operator/=(const T x)
   {
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] /= x;
   return *this;
   }

// user-defined operations

template <class T>
inline masked_vector<T>& masked_vector<T>::apply(T f(T))
   {
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         m_data->m_data[i] = f(m_data->m_data[i]);
   return *this;
   }

// information services

template <class T>
inline int masked_vector<T>::size() const
   {
   int result = 0;
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         result++;
   return result;
   }

// statistical operations

template <class T>
inline T masked_vector<T>::min() const
   {
   assert(size() > 0);
   T result;
   bool initial = true;
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i) && (m_data->m_data[i] < result || initial))
         {
         result = m_data->m_data[i];
         initial = false;
         }
   return result;
   }

template <class T>
inline T masked_vector<T>::max() const
   {
   assert(size() > 0);
   T result;
   bool initial = true;
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i) && (m_data->m_data[i] > result || initial))
         {
         result = m_data->m_data[i];
         initial = false;
         }
   return result;
   }

template <class T>
inline T masked_vector<T>::sum() const
   {
   assert(m_data->m_size.length() > 0);
   T result = 0;
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         result += m_data->m_data[i];
   return result;
   }

template <class T>
inline T masked_vector<T>::sumsq() const
   {
   assert(m_data->m_size.length() > 0);
   T result = 0;
   for (int i = 0; i < m_data->m_size.length(); i++)
      if (m_mask(i))
         result += m_data->m_data[i] * m_data->m_data[i];
   return result;
   }

template <class T>
inline T masked_vector<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq() / T(size()) - _mean * _mean;
   return (_var > 0) ? _var : 0;
   }

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
