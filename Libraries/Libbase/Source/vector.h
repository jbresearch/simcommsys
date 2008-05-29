#ifndef __vector_h
#define __vector_h

#include "config.h"
#include <stdlib.h>
#include <iostream>

namespace libbase {

/*!
   \brief   Generic Vector.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (29 Oct 2001)
   reinstated arithmetic functions as part of the vector class. These are only created
   for an instantiation in which they are used, so it should not pose a problem anyway.
   Also, cleaned up the protected (internal) interface, and formalised the existence of
   empty vectors.

   \version 1.11 (31 Oct 2001)
   renamed member variables to start with m_ to enable similar naming of public functions.
   This is particularly useful for matrices. Also solved a bug where vector size was not
   checked for validity in new vectors, and created a new internal function to validate
   that the vector is not empty.

   \version 1.20 (11 Nov 2001)
   renamed max/min functions to max/min, after #undef'ing the macros with that name in
   Win32 (and possibly other compilers). Also added a new function to compute the sum
   of elements in a vector.

   \version 1.30 (30 Nov 2001)
   added statistical functions that return sumsq, mean, var.

   \version 1.31 (2 Dec 2001)
   added a function which sets the size of a vector to the given size - leaving it as
   it is if the size was already good, and freeing/reallocating if necessary. This helps
   reduce redundant free/alloc operations on matrices which keep the same size.
   [Ported from matrix 1.11]

   \version 1.40 (27 Feb 2002)
   modified the stream output function to first print the size, and added a complementary
   stream input function. Together these allow for simplified saving and loading.

   \version 1.41 (6 Mar 2002)
   changed use of iostream from global to std namespace.

   \version 1.42 (4 Apr 2002)
   made validation functions operative only in debug mode.

   \version 1.43 (7 Apr 2002)
   moved validation functions up-front, to make sure they're used inline. Also moved
   alloc and free before setsize, for the same reason.

   \version 1.50 (13 Apr 2002)
   added a number of high-level support routines for working with vectors - the overall
   effect of this should be a drastic reduction in the number of loops required in user
   code to express various common operations. Changes are:
   - support for working with different-sized vectors (in place of resizing operations
   which would be quite expensive); added a function copyfrom() which copies data from
   another vector without resizing this one. Opted for this rather than changing the
   definition of operator= because it's convenient for '=' to copy _everything_ from the
   source to the destination; otherwise we would land into obscure problems in some cases
   (like when we're trying to copy a vector of matrices, etc.). This method also has the
   advantage of keeping the old code/interface as it was.
   - added a new format for init(), which takes another vector as argument, to allow
   easier (and neater) sizing of one vector based on another. This is a template function
   to allow the argument vector to be of a different type.
   - added an apply() function which allows the user to do the same operation on all
   elements (previously had to do this manually).

   \version 1.51 (13 Apr 2002)
   removed all validate functions & replaced them by assertions.

   \version 1.52 (15 Apr 2002)
   fixed a bug in sum & sumsq (and by consequence mean, var & sigma) - the first element
   was being skipped in the loop.

   \version 1.60 (30 Apr 2002)
   added assign() function to copy elements from an array.

   \version 1.61 (9 May 2002)
   - added another apply() so that the given function's parameter is const - this allows
   the use of functions which do not modify their parameter (it's actually what we want
   anyway). The older version (with non-const parameter) is still kept to allow the use
   of functions where the parameter is not defined as const (such as fabs, etc).
   - added unary and binary boolean operators.
   - also, changed the binary operators to be member functions with a single argument,
   rather than non-members with two arguments. Also, for operations with a constant
   (rather than another vector), that constant is passed directly, not by reference.
   - added serialize() functions which read and write vector data only; the input function
   assumes that the current vector already has the correct size. These functions are
   useful for interfacing with other file formats. Also modified the stream I/O functions
   to make use of these.

   \version 1.62 (11 Jun 2002)
   removed the instance of apply() whose given function's parameter is const, since this
   was causing problems with gcc on Solaris.

   \version 1.63 (5 Jan 2005)
   fixed the templated init function that takes a vector as parameter, so that the size
   is obtained through the respective function (instead of by directly tyring to read the
   private member variables). This is to allow this function to be given as parameter a
   vector of different type. [as fixed in matrix v1.64]

   \version 1.70 (5 Jan 2005)
   - added alternative min() and max() functions which take a dereferenced integer as
   parameter, to allow the function to return the index for the min or max value,
   respectively. A second optional parameter allows the user to obtain the index for
   the first or the last min/max value (defaults to first).

   \version 1.71 (18 Jul 2006)
   updated declaration of vector's friend functions to comply with the standard, by
   adding declarations of the function before that of the class. Consequently, a
   declaration of the class itself was also required before that.

   \version 1.72 (6 Oct 2006)
   renamed GCCONLY to STRICT, in accordance with config 2.07.

   \version 1.73 (7 Oct 2006)
   renamed STRICT to TPLFRIEND, in accordance with config 2.08.

   \version 1.74 (13 Oct 2006)
   removed TPLFRIEND, in accordance with config 3.00.

   \version 1.80 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.81 (17 Oct 2007)
   - modified alloc() so that m_data is set to NULL if we're not allocating space; this silences a warning.

   \version 1.90 (15 Nov 2007)
   - added facility for sub-vector referencing, enabling access to the sub-vector data without
    array copying; this needed the introduction of a flag m_root, which indicates vectors
    containing their own allocated memory.

   \version 1.91 (28 Nov 2007)
   - defined alternate vector copy for non-root vectors (to avoid copying the data)

   \version 2.00 (4-6 Jan 2008)
   - hid vector multiplication and division as private functions to make sure they
     are not being used anywhere.
   - updated allocator to detect invalid size values

     
   \todo This class needs to be re-designed in a manner that is consistent with
         convention (esp. Matlab) and that is efficient
*/

template <class T> class vector;

template <class T> std::ostream& operator<<(std::ostream& s, const vector<T>& x);
template <class T> std::istream& operator>>(std::istream& s, vector<T>& x);

template <class T> class vector {
protected:
   bool  m_root;
   int   m_xsize;
   T     *m_data;
protected:
   // memory allocation functions
   void alloc(const int x);   // allocates memory for x elements (if necessary) and updates xsize
   void free();               // if there is memory allocated, free it
   void setsize(const int x); // set vector to given size, freeing if and as required
public:
   vector(const int x=0) { alloc(x); };  // constructor (does not initialise elements)
   vector(const vector<T>& x);
   ~vector() { free(); };

   // resizing operations
   void init(const int x) { setsize(x); };
   template <class A> void init(const vector<A>& x) { init(x.size()); };

   // vector copy and value initialisation
   vector<T>& assign(const T* x, const int n);
   vector<T>& copyfrom(const vector<T>& x);
   vector<T>& operator=(const vector<T>& x);
   vector<T>& operator=(const T x);

   // sub-vector access
   const vector<T> extract(const int start, const int n) const;

   // index operators (perform boundary checking)
   T& operator()(const int x);
   T operator()(const int x) const;

   // information services
   int size() const { return m_xsize; };                 //!< Total number of elements

   // serialization and stream input & output
   void serialize(std::ostream& s) const;
   void serialize(std::istream& s);
   friend std::ostream& operator<< <>(std::ostream& s, const vector<T>& x);
   friend std::istream& operator>> <>(std::istream& s, vector<T>& x);

   // arithmetic operations - unary
   vector<T>& operator+=(const vector<T>& x);
   vector<T>& operator-=(const vector<T>& x);
private:
   vector<T>& operator*=(const vector<T>& x);
   vector<T>& operator/=(const vector<T>& x);
public:
   vector<T>& operator+=(const T x);
   vector<T>& operator-=(const T x);
   vector<T>& operator*=(const T x);
   vector<T>& operator/=(const T x);

   // arithmetic operations - binary
   vector<T> operator+(const vector<T>& x) const;
   vector<T> operator-(const vector<T>& x) const;
private:
   vector<T> operator*(const vector<T>& x) const;
   vector<T> operator/(const vector<T>& x) const;
public:
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

   // user-defined operations
   vector<T>& apply(T f(T));

   // statistical operations
   T min() const;
   T max() const;
   T min(int& index, const bool getfirst=true) const;
   T max(int& index, const bool getfirst=true) const;
   T sum() const;
   T sumsq() const;
   T mean() const { return sum()/T(size()); };
   T var() const;
   T sigma() const { return sqrt(var()); };
};

// memory allocation functions

template <class T> inline void vector<T>::alloc(const int x)
   {
   assert(x >= 0);
   m_xsize = x;
   m_root = true;
   if(x > 0)
      m_data = new T[x];
   else
      m_data = NULL;
   }

template <class T> inline void vector<T>::free()
   {
   if(m_root && m_xsize > 0)
      delete[] m_data;
   }

template <class T> inline void vector<T>::setsize(const int x)
   {
   assert(x >= 0);
   if(x==m_xsize)
      return;
   free();
   alloc(x);
   }

// constructor / destructor functions

template <class T> inline vector<T>::vector(const vector<T>& x)
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

template <class T> inline vector<T>& vector<T>::assign(const T* x, const int n)
   {
   setsize(n);
   for(int i=0; i<n; i++)
      m_data[i] = x[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::copyfrom(const vector<T>& x)
   {
   const int xsize = ::min(m_xsize, x.m_xsize);
   for(int i=0; i<xsize; i++)
      m_data[i] = x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator=(const vector<T>& x)
   {
   setsize(x.m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] = x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = x;
   return *this;
   }

// sub-vector access

template <class T> inline const vector<T> vector<T>::extract(const int start, const int n) const
   {
   vector<T> r;
   r.m_root = false;
   r.m_xsize = n;
   assert(m_xsize >= start+n);
   r.m_data = &m_data[start];
   return r;
   }

// index operators (perform boundary checking)

template <class T> inline T& vector<T>::operator()(const int x)
   {
   assert(x>=0 && x<m_xsize);
   return m_data[x];
   }

template <class T> inline T vector<T>::operator()(const int x) const
   {
   assert(x>=0 && x<m_xsize);
   return m_data[x];
   }

// serialization and stream input & output

template <class T> inline void vector<T>::serialize(std::ostream& s) const
   {
   s << m_data[0];
   for(int i=1; i<m_xsize; i++)
      s << "\t" << m_data[i];
   s << "\n";
   }

template <class T> inline void vector<T>::serialize(std::istream& s)
   {
   for(int i=0; i<m_xsize; i++)
      s >> m_data[i];
   }

template <class T> inline std::ostream& operator<<(std::ostream& s, const vector<T>& x)
   {
   s << x.m_xsize << "\n";
   x.serialize(s);
   return s;
   }

template <class T> inline std::istream& operator>>(std::istream& s, vector<T>& x)
   {
   int xsize;
   s >> xsize;
   x.setsize(xsize);
   x.serialize(s);
   return s;
   }

// arithmetic operations - unary

template <class T> inline vector<T>& vector<T>::operator+=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] += x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator-=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] -= x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator*=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] *= x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator/=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] /= x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator+=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] += x;
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator-=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] -= x;
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator*=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] *= x;
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator/=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] /= x;
   return *this;
   }

// arithmetic operations - binary

template <class T> inline vector<T> vector<T>::operator+(const vector<T>& x) const
   {
   vector<T> r = *this;
   r += x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator-(const vector<T>& x) const
   {
   vector<T> r = *this;
   r -= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator*(const vector<T>& x) const
   {
   vector<T> r = *this;
   r *= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator/(const vector<T>& x) const
   {
   vector<T> r = *this;
   r /= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator+(const T x) const
   {
   vector<T> r = *this;
   r += x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator-(const T x) const
   {
   vector<T> r = *this;
   r -= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator*(const T x) const
   {
   vector<T> r = *this;
   r *= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator/(const T x) const
   {
   vector<T> r = *this;
   r /= x;
   return r;
   }

// boolean operations - unary

template <class T> inline vector<T>& vector<T>::operator!()
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = !m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator&=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] &= x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator|=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] |= x.m_data[i];
   return *this;
   }

template <class T> inline vector<T>& vector<T>::operator^=(const vector<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   for(int i=0; i<m_xsize; i++)
      m_data[i] ^= x.m_data[i];
   return *this;
   }

// boolean operations - binary

template <class T> inline vector<T> vector<T>::operator&(const vector<T>& x) const
   {
   vector<T> r = *this;
   r &= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator|(const vector<T>& x) const
   {
   vector<T> r = *this;
   r |= x;
   return r;
   }

template <class T> inline vector<T> vector<T>::operator^(const vector<T>& x) const
   {
   vector<T> r = *this;
   r ^= x;
   return r;
   }

// user-defined operations

template <class T> inline vector<T>& vector<T>::apply(T f(T))
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = f(m_data[i]);
   return *this;
   }

// statistical operations

template <class T> inline T vector<T>::min() const
   {
   assert(m_xsize > 0);
   T result = m_data[0];
   for(int i=1; i<m_xsize; i++)
      if(m_data[i] < result)
         result = m_data[i];
   return result;
   }

template <class T> inline T vector<T>::max() const
   {
   assert(m_xsize > 0);
   T result = m_data[0];
   for(int i=1; i<m_xsize; i++)
      if(m_data[i] > result)
         result = m_data[i];
   return result;
   }

template <class T> inline T vector<T>::min(int& index, const bool getfirst) const
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

template <class T> inline T vector<T>::max(int& index, const bool getfirst) const
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

template <class T> inline T vector<T>::sum() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      result += m_data[i];
   return result;
   }

template <class T> inline T vector<T>::sumsq() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      result += m_data[i] * m_data[i];
   return result;
   }

template <class T> inline T vector<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

}; // end namespace

#endif
