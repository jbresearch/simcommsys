#ifndef __matrix3_h
#define __matrix3_h

#include "config.h"
#include <stdlib.h>
#include <iostream>

namespace libbase {

/*!
   \brief   Generic 3D Matrix.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00 (31 Oct 2001)
  separated 3D matrix class from the 2D matrix header.
  Created arithmetic functions as part of the matrix class. These are only created
  for an instantiation in which they are used, so it should not pose a problem anyway.
  This includes arithmetic operations between matrices, constant matrix initialisation
  routines, and also some statistical functions. Also, cleaned up the protected
  (internal) interface, and formalised the existence of empty matrices. Also renamed
  member variables to start with m_ in order to facilitate the similar naming of public
  functions (such as the size functions).

  Version 1.10 (11 Nov 2001)
  renamed max/min functions to max/min, after #undef'ing the macros with that name in
  Win32 (and possibly other compilers). Also added a new function to compute the sum
  of elements in a matrix.

  Version 1.20 (30 Nov 2001)
  added statistical functions that return sumsq, mean, var.

  Version 1.21 (2 Dec 2001)
  added a function which sets the size of a matrix to the given size - leaving it as
  it is if the size was already good, and freeing/reallocating if necessary. This helps
  reduce redundant free/alloc operations on matrices which keep the same size.
  [Ported from matrix 1.11]

  Version 1.30 (27 Feb 2002)
  modified the stream output function to first print the size, and added a complementary
  stream input function. Together these allow for simplified saving and loading.

  Version 1.31 (6 Mar 2002)
  changed use of iostream from global to std namespace.

  Version 1.32 (4 Apr 2002)
  made validation functions operative only in debug mode.

  Version 1.33 (7 Apr 2002)
  moved validation functions up-front, to make sure they're used inline. Also moved
  alloc and free before setsize, for the same reason.

  Version 1.40 (13 Apr 2002)
  skipped for version consistency with matrix & vector.

  Version 1.50 (13 Apr 2002)
  added a number of high-level support routines for working with matrices - the overall
  effect of this should be a drastic reduction in the number of loops required in user
  code to express various common operations. Changes are:
  * support for working with different-sized matrices (in place of resizing operations
  which would be quite expensive); added a function copyfrom() which copies data from
  another matrix without resizing this one. Opted for this rather than changing the
  definition of operator= because it's convenient for '=' to copy _everything_ from the
  source to the destination; otherwise we would land into obscure problems in some cases
  (like when we're trying to copy a vector of matrices, etc.). This method also has the
  advantage of keeping the old code/interface as it was.
  * added a new format for init(), which takes another matrix as argument, to allow
  easier (and neater) sizing of one matrix based on another. This is a template function
  to allow the argument matrix to be of a different type.
  * added an apply() function which allows the user to do the same operation on all
  elements (previously had to do this manually).

  Version 1.51 (13 Apr 2002)
  removed all validate functions & replaced them by assertions.

  Version 1.60 (9 May 2002)
  * added another apply() so that the given function's parameter is const - this allows
  the use of functions which do not modify their parameter (it's actually what we want
  anyway). The older version (with non-const parameter) is still kept to allow the use
  of functions where the parameter is not defined as const (such as fabs, etc).
  * added unary and binary boolean operators.
  * also, changed the binary operators to be member functions with a single argument,
  rather than non-members with two arguments. Also, for operations with a constant
  (rather than another vector), that constant is passed directly, not by reference.
  * added serialize() functions which read and write vector data only; the input function
  assumes that the current vector already has the correct size. These functions are
  useful for interfacing with other file formats. Also modified the stream I/O functions
  to make use of these.

  Version 1.61 (11 Jun 2002)
  removed the instance of apply() whose given function's parameter is const, since this
  was causing problems with gcc on Solaris.

  Version 1.62 (18 Jul 2006)
  updated declaration of matrix3's friend functions to comply with the standard, by
  adding declarations of the function before that of the class. Consequently, a
  declaration of the class itself was also required before that.

  Version 1.63 (6 Oct 2006)
  renamed GCCONLY to STRICT, in accordance with config 2.07.

  Version 1.64 (7 Oct 2006)
  renamed STRICT to TPLFRIEND, in accordance with config 2.08.

  Version 1.65 (13 Oct 2006)
  removed TPLFRIEND, in accordance with config 3.00.

  Version 1.70 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.71 (17 Oct 2007)
  * modified alloc() so that m_data is set to NULL if we're not allocating space; this silences a warning.
*/

template <class T> class matrix3;

template <class T> std::ostream& operator<<(std::ostream& s, const matrix3<T>& x);
template <class T> std::istream& operator>>(std::istream& s, matrix3<T>& x);

template <class T> class matrix3 {
private:
   int  m_xsize, m_ysize, m_zsize;
   T    ***m_data;
protected:
   // memory allocation functions
   void alloc(const int x, const int y, const int z);   // allocates memory for (x,y,z) elements and updates sizes
   void free();                            // if there is memory allocated, free it
   void setsize(const int x, const int y, const int z); // set matrix to given size, freeing if and as required
public:
   matrix3(const int x=0, const int y=0, const int z=0);  // constructor (does not initialise elements)
   matrix3(const matrix3<T>& x);                          // copy constructor
   ~matrix3();

   // resizing operations
   void init(const int x, const int y, const int z);
   template <class A> void init(const matrix3<A>& x) { init(x.m_xsize, x.m_ysize, x.m_zsize); };

   // matrix3 copy and value initialisation
   matrix3<T>& copyfrom(const matrix3<T>& x);
   matrix3<T>& operator=(const matrix3<T>& x);
   matrix3<T>& operator=(const T x);

   // index operators (perform boundary checking)
   T& operator()(const int x, const int y, const int z);
   T operator()(const int x, const int y, const int z) const;

   // information services
   int xsize() const { return m_xsize; };          // size on dimension x
   int ysize() const { return m_ysize; };          // size on dimension y
   int zsize() const { return m_zsize; };          // size on dimension y
   int size() const { return m_xsize * m_ysize * m_zsize; }; // total number of elements

   // serialization and stream input & output
   void serialize(std::ostream& s) const;
   void serialize(std::istream& s);
   friend std::ostream& operator<< <>(std::ostream& s, const matrix3<T>& x);
   friend std::istream& operator>> <>(std::istream& s, matrix3<T>& x);

   // arithmetic operations - unary
   matrix3<T>& operator+=(const matrix3<T>& x);
   matrix3<T>& operator-=(const matrix3<T>& x);
   matrix3<T>& operator*=(const matrix3<T>& x);
   matrix3<T>& operator/=(const matrix3<T>& x);
   matrix3<T>& operator+=(const T x);
   matrix3<T>& operator-=(const T x);
   matrix3<T>& operator*=(const T x);
   matrix3<T>& operator/=(const T x);

   // arithmetic operations - binary
   matrix3<T> operator+(const matrix3<T>& x) const;
   matrix3<T> operator-(const matrix3<T>& x) const;
   matrix3<T> operator*(const matrix3<T>& x) const;
   matrix3<T> operator/(const matrix3<T>& x) const;
   matrix3<T> operator+(const T x) const;
   matrix3<T> operator-(const T x) const;
   matrix3<T> operator*(const T x) const;
   matrix3<T> operator/(const T x) const;

   // boolean operations - unary
   matrix3<T>& operator!();
   matrix3<T>& operator&=(const matrix3<T>& x);
   matrix3<T>& operator|=(const matrix3<T>& x);
   matrix3<T>& operator^=(const matrix3<T>& x);

   // boolean operations - binary
   matrix3<T> operator&(const matrix3<T>& x) const;
   matrix3<T> operator|(const matrix3<T>& x) const;
   matrix3<T> operator^(const matrix3<T>& x) const;

   // user-defined operations
   matrix3<T>& apply(T f(T));

   // statistical operations
   T min() const;
   T max() const;
   T sum() const;
   T sumsq() const;
   T mean() const { return sum()/T(size()); };
   T var() const;
   T sigma() const { return sqrt(var()); };
};

// memory allocation functions

template <class T> inline void matrix3<T>::alloc(const int x, const int y, const int z)
   {
   if(x==0 || y==0 || z==0)
      {
      m_xsize = 0;
      m_ysize = 0;
      m_zsize = 0;
      m_data = NULL;
      }
   else
      {
      m_xsize = x;
      m_ysize = y;
      m_zsize = z;
      typedef T** Tpp;
      typedef T* Tp;
      m_data = new Tpp[x];
      for(int i=0; i<x; i++)
         {
         m_data[i] = new Tp[y];
         for(int j=0; j<y; j++)
            m_data[i][j] = new T[z];
         }
      }
   }

template <class T> inline void matrix3<T>::free()
   {
   // note that if xsize is 0, then ysize & zsize must also be zero
   if(m_xsize > 0)
      {
      for(int i=0; i<m_xsize; i++)
         {
         for(int j=0; j<m_ysize; j++)
            delete[] m_data[i][j];
         delete[] m_data[i];
         }
      delete[] m_data;
      }
   }

template <class T> inline void matrix3<T>::setsize(const int x, const int y, const int z)
   {
   if(x==m_xsize && y==m_ysize && z==m_zsize)
      return;
   free();
   alloc(x,y,z);
   }

// constructor / destructor functions

template <class T> inline matrix3<T>::matrix3(const int x, const int y, const int z)
   {
   assert(x >= 0);
   assert(y >= 0);
   assert(z >= 0);
   alloc(x,y,z);
   }

template <class T> inline matrix3<T>::matrix3(const matrix3<T>& x)
   {
   alloc(x.m_xsize, x.m_ysize, x.m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] = x.m_data[i][j][k];
   }

template <class T> inline matrix3<T>::~matrix3()
   {
   free();
   }

// resizing operations

template <class T> inline void matrix3<T>::init(const int x, const int y, const int z)
   {
   assert(x >= 0);
   assert(y >= 0);
   assert(z >= 0);
   setsize(x,y,z);
   }

// matrix copy and value initialisation

template <class T> inline matrix3<T>& matrix3<T>::copyfrom(const matrix3<T>& x)
   {
   const int xsize = ::min(m_xsize, x.m_xsize);
   const int ysize = ::min(m_ysize, x.m_ysize);
   const int zsize = ::min(m_zsize, x.m_zsize);
   for(int i=0; i<xsize; i++)
      for(int j=0; j<ysize; j++)
         for(int k=0; k<zsize; k++)
            m_data[i][j][k] = x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator=(const matrix3<T>& x)
   {
   setsize(x.m_xsize, x.m_ysize, x.m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] = x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] = x;
   return *this;
   }

// index operators (perform boundary checking)

template <class T> inline T& matrix3<T>::operator()(const int x, const int y, const int z)
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   assert(z>=0 && z<m_zsize);
   return m_data[x][y][z];
   }

template <class T> inline T matrix3<T>::operator()(const int x, const int y, const int z) const
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   assert(z>=0 && z<m_zsize);
   return m_data[x][y][z];
   }

// serialization and stream input & output

template <class T> inline void matrix3<T>::serialize(std::ostream& s) const
   {
   for(int k=0; k<m_zsize; k++)
      {
      s << "\n";
      for(int j=0; j<m_ysize; j++)
         {
         s << m_data[0][j][k];
         for(int i=1; i<m_xsize; i++)
            s << "\t" << m_data[i][j][k];
         s << "\n";
         }
      }
   }

template <class T> inline void matrix3<T>::serialize(std::istream& s)
   {
   for(int k=0; k<m_zsize; k++)
      for(int j=0; j<m_ysize; j++)
         for(int i=0; i<m_xsize; i++)
            s >> m_data[i][j][k];
   }

template <class T> inline std::ostream& operator<<(std::ostream& s, const matrix3<T>& x)
   {
   s << x.m_xsize << "\t" << x.m_ysize << "\t" << x.m_zsize << "\n";
   x.serialize(s);
   return s;
   }

template <class T> inline std::istream& operator>>(std::istream& s, matrix3<T>& x)
   {
   int xsize, ysize, zsize;
   s >> xsize >> ysize >> zsize;
   x.setsize(xsize,ysize,zsize);
   x.serialize(s);
   return s;
   }

// arithmetic operations - unary

template <class T> inline matrix3<T>& matrix3<T>::operator+=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] += x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator-=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] -= x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator*=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] *= x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator/=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] /= x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator+=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] += x;
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator-=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] -= x;
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator*=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] *= x;
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator/=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] /= x;
   return *this;
   }

// arithmetic operations - binary

template <class T> inline matrix3<T> matrix3<T>::operator+(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r += x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator-(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r -= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator*(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r *= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator/(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r /= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator+(const T x) const
   {
   matrix3<T> r = *this;
   r += x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator-(const T x) const
   {
   matrix3<T> r = *this;
   r -= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator*(const T x) const
   {
   matrix3<T> r = *this;
   r *= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator/(const T x) const
   {
   matrix3<T> r = *this;
   r /= x;
   return r;
   }

// boolean operations - unary

template <class T> inline matrix3<T>& matrix3<T>::operator!()
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] = !m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator&=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] &= x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator|=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] |= x.m_data[i][j][k];
   return *this;
   }

template <class T> inline matrix3<T>& matrix3<T>::operator^=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] ^= x.m_data[i][j][k];
   return *this;
   }

// boolean operations - binary

template <class T> inline matrix3<T> matrix3<T>::operator&(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r &= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator|(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r |= x;
   return r;
   }

template <class T> inline matrix3<T> matrix3<T>::operator^(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r ^= x;
   return r;
   }

// user-defined operations

template <class T> inline matrix3<T>& matrix3<T>::apply(T f(T))
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            m_data[i][j][k] = f(m_data[i][j][k]);
   return *this;
   }

// statistical operations

template <class T> inline T matrix3<T>::min() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0][0];
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            if(m_data[i][j][k] < result)
               result = m_data[i][j][k];
   return result;
   }

template <class T> inline T matrix3<T>::max() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0][0];
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            if(m_data[i][j][k] > result)
               result = m_data[i][j][k];
   return result;
   }

template <class T> inline T matrix3<T>::sum() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            result += m_data[i][j][k];
   return result;
   }

template <class T> inline T matrix3<T>::sumsq() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         for(int k=0; k<m_zsize; k++)
            result += m_data[i][j][k] * m_data[i][j][k];
   return result;
   }

template <class T> inline T matrix3<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

}; // end namespace

#endif
