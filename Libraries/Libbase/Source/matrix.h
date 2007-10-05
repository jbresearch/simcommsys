#ifndef __matrix_h
#define __matrix_h

#include "config.h"
#include "vcs.h"
#include "vector.h"
#include <stdlib.h>
#include <iostream>

/*
  Version 1.10 (31 Oct 2001)
  created arithmetic functions as part of the matrix class. These are only created
  for an instantiation in which they are used, so it should not pose a problem anyway.
  This includes arithmetic operations between matrices, constant matrix initialisation
  routines, and also some statistical functions. Also, cleaned up the protected
  (internal) interface, and formalised the existence of empty matrices. Also renamed
  member variables to start with m_ in order to facilitate the similar naming of public
  functions (such as the size functions). Separated 3D matrix into another header.

  Version 1.11 (10 Nov 2001)
  added a function which sets the size of a matrix to the given size - leaving it as
  it is if the size was already good, and freeing/reallocating if necessary. This helps
  reduce redundant free/alloc operations on matrices which keep the same size.

  Version 1.20 (11 Nov 2001)
  renamed max/min functions to max/min, after #undef'ing the macros with that name in
  Win32 (and possibly other compilers). Also added a new function to compute the sum
  of elements in a matrix.

  Version 1.30 (30 Nov 2001)
  added statistical functions that return sumsq, mean, var.

  Version 1.40 (27 Feb 2002)
  modified the stream output function to first print the size, and added a complementary
  stream input function. Together these allow for simplified saving and loading.

  Version 1.41 (6 Mar 2002)
  changed use of iostream from global to std namespace.

  Version 1.42 (4 Apr 2002)
  made validation functions operative only in debug mode.

  Version 1.43 (7 Apr 2002)
  moved validation functions up-front, to make sure they're used inline. Also moved
  alloc and free before setsize, for the same reason.

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
  * added functions to extract/insert matrix rows and columns as vectors - for the moment
  these need the target vector to be passed as a parameter; the expression format can
  be improved aesthetically, however the present format clearly communicates what is
  happening.

  Version 1.51 (13 Apr 2002)
  removed all validate functions & replaced them by assertions.

  Version 1.60 (15 Apr 2002)
  added functions to support masked operations:
  * added mask creation functions by defining comparison operators.
  * added a new class masked_matrix, created by masking any matrix, and defined the
  arithmetic, statistical, user-defined operation, and copy/value init functions for this
  class. This easily allows us to modify the masked parts of any given matrix. Also
  note here that for the user, the use of masked matrices should be essentially
  transparent (in that they can mostly be used in place of normal matrices) and that
  the user should never create one explicitly, but merely through a normal matrix.
  * added unary and binary boolean operators, for use with matrix<bool> mostly.
  * also, changed the binary operators to be member functions with a single argument,
  rather than non-members with two arguments.
  * finally, added conversion from matrix (or masked matrix) to vector

  Version 1.61 (22 Apr 2002)
  added serialize() functions which read and write matrix data only; the input function
  assumes that the current matrix already has the correct size. These functions are
  useful for interfacing with other file formats. Also modified the stream I/O functions
  to make use of these.

  Version 1.62 (9 May 2002)
  * added another apply() so that the given function's parameter is const - this allows
  the use of functions which do not modify their parameter (it's actually what we want
  anyway). The older version (with non-const parameter) is still kept to allow the use
  of functions where the parameter is not defined as const (such as fabs, etc).

  Version 1.63 (11 Jun 2002)
  removed the instance of apply() whose given function's parameter is const, since this
  was causing problems with gcc on Solaris.

  Version 1.64 (5 Jan 2005)
  fixed the templated init function that takes a matrix as parameter, so that the xsize
  and ysize are obtained through the respective functions (instead of by directly tyring
  to read the private member variables). This is to allow this function to be given as
  parameter a matrix of different type.

  Version 1.65 (18 Jul 2006)
  updated declaration of matrix's friend functions to comply with the standard, by
  adding declarations of the function before that of the class. Consequently, a
  declaration of the class itself was also required before that.

  Version 1.66 (6 Oct 2006)
  renamed GCCONLY to STRICT, in accordance with config 2.07.

  Version 1.67 (7 Oct 2006)
  renamed STRICT to TPLFRIEND, in accordance with config 2.08.

  Version 1.68 (13 Oct 2006)
  removed TPLFRIEND, in accordance with config 3.00.

  Version 1.70 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libbase {

extern const vcs matrix_version;

template <class T> class masked_matrix;
template <class T> class matrix;

template <class T> std::ostream& operator<<(std::ostream& s, const matrix<T>& x);
template <class T> std::istream& operator>>(std::istream& s, matrix<T>& x);

template <class T> class matrix {
   friend class masked_matrix<T>;
private:
   int  m_xsize, m_ysize;
   T    **m_data;
protected:
   // memory allocation functions
   void alloc(const int x, const int y);   // allocates memory for (x,y) elements and updates sizes
   void free();                            // if there is memory allocated, free it
   void setsize(const int x, const int y); // set matrix to given size, freeing if and as required
public:
   matrix(const int x=0, const int y=0);        // constructor (does not initialise elements)
   matrix(const matrix<T>& m);                  // copy constructor
   ~matrix();

   // resizing operations
   void init(const int x, const int y);
   template <class A> void init(const matrix<A>& x) { init(x.xsize(), x.ysize()); };

   // matrix copy and value initialisation
   matrix<T>& copyfrom(const matrix<T>& x);
   matrix<T>& operator=(const matrix<T>& x);
   matrix<T>& operator=(const T x);

   // insert/extract rows/columns as vectors
   void insertrow(const vector<T>& v, const int x);
   void extractrow(vector<T>& v, const int x) const;
   void insertcol(const vector<T>& v, const int y);
   void extractcol(vector<T>& v, const int y) const;

   // convert to a vector
   operator vector<T>() const;

   // bind a mask to a matrix
   masked_matrix<T> mask(const matrix<bool>& m) { return masked_matrix<T>(this, m); };

   // index operators (perform boundary checking)
   T& operator()(const int x, const int y);
   T operator()(const int x, const int y) const;

   // information services
   int xsize() const { return m_xsize; };          // size on dimension x
   int ysize() const { return m_ysize; };          // size on dimension y
   int size() const { return m_xsize * m_ysize; }; // total number of elements

   // serialization and stream input & output
   void serialize(std::ostream& s) const;
   void serialize(std::istream& s);
   friend std::ostream& operator<< <>(std::ostream& s, const matrix<T>& x);
   friend std::istream& operator>> <>(std::istream& s, matrix<T>& x);

   // comparison (mask-creation) operations
   matrix<bool> operator==(const matrix<T>& x) const;
   matrix<bool> operator!=(const matrix<T>& x) const;
   matrix<bool> operator<=(const matrix<T>& x) const;
   matrix<bool> operator>=(const matrix<T>& x) const;
   matrix<bool> operator<(const matrix<T>& x) const;
   matrix<bool> operator>(const matrix<T>& x) const;
   matrix<bool> operator==(const T x) const;
   matrix<bool> operator!=(const T x) const;
   matrix<bool> operator<=(const T x) const;
   matrix<bool> operator>=(const T x) const;
   matrix<bool> operator<(const T x) const;
   matrix<bool> operator>(const T x) const;

   // arithmetic operations - unary
   matrix<T>& operator+=(const matrix<T>& x);
   matrix<T>& operator-=(const matrix<T>& x);
   matrix<T>& operator*=(const matrix<T>& x);
   matrix<T>& operator/=(const matrix<T>& x);
   matrix<T>& operator+=(const T x);
   matrix<T>& operator-=(const T x);
   matrix<T>& operator*=(const T x);
   matrix<T>& operator/=(const T x);

   // arithmetic operations - binary
   matrix<T> operator+(const matrix<T>& x) const;
   matrix<T> operator-(const matrix<T>& x) const;
   matrix<T> operator*(const matrix<T>& x) const;
   matrix<T> operator/(const matrix<T>& x) const;
   matrix<T> operator+(const T x) const;
   matrix<T> operator-(const T x) const;
   matrix<T> operator*(const T x) const;
   matrix<T> operator/(const T x) const;

   // boolean operations - unary
   matrix<T>& operator!();
   matrix<T>& operator&=(const matrix<T>& x);
   matrix<T>& operator|=(const matrix<T>& x);
   matrix<T>& operator^=(const matrix<T>& x);

   // boolean operations - binary
   matrix<T> operator&(const matrix<T>& x) const;
   matrix<T> operator|(const matrix<T>& x) const;
   matrix<T> operator^(const matrix<T>& x) const;

   // user-defined operations
   matrix<T>& apply(T f(T));

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

template <class T> inline void matrix<T>::free()
   {
   // note that if xsize is 0, then ysize must also be zero
   if(m_xsize > 0)
      {
      for(int i=0; i<m_xsize; i++)
         delete[] m_data[i];
      delete[] m_data;
      }
   }

template <class T> inline void matrix<T>::setsize(const int x, const int y)
   {
   if(x==m_xsize && y==m_ysize)
      return;
   free();
   alloc(x,y);
   }

template <class T> inline void matrix<T>::alloc(const int x, const int y)
   {
   if(x==0 || y==0)
      {
      m_xsize = 0;
      m_ysize = 0;
      }
   else
      {
      m_xsize = x;
      m_ysize = y;
      typedef T* Tp;
      m_data = new Tp[x];
      for(int i=0; i<x; i++)
         m_data[i] = new T[y];
      }
   }

// constructor / destructor functions

template <class T> inline matrix<T>::matrix(const int x, const int y)
   {
   assert(x >= 0);
   assert(y >= 0);
   alloc(x,y);
   }

template <class T> inline matrix<T>::matrix(const matrix<T>& m)
   {
   alloc(m.m_xsize, m.m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = m.m_data[i][j];
   }

template <class T> inline matrix<T>::~matrix()
   {
   free();
   }

// resizing operations

template <class T> inline void matrix<T>::init(const int x, const int y)
   {
   assert(x >= 0);
   assert(y >= 0);
   setsize(x,y);
   }

// matrix copy and value initialisation

template <class T> inline matrix<T>& matrix<T>::copyfrom(const matrix<T>& x)
   {
   const int xsize = ::min(m_xsize, x.m_xsize);
   const int ysize = ::min(m_ysize, x.m_ysize);
   for(int i=0; i<xsize; i++)
      for(int j=0; j<ysize; j++)
         m_data[i][j] = x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator=(const matrix<T>& x)
   {
   setsize(x.m_xsize, x.m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = x;
   return *this;
   }

// insert/extract rows/columns as vectors

template <class T> inline void matrix<T>::insertrow(const vector<T>& v, const int x)
   {
   assert(v.size() == m_ysize);
   for(int y=0; y<m_ysize; y++)
      m_data[x][y] = v(y);
   }

template <class T> inline void matrix<T>::extractrow(vector<T>& v, const int x) const
   {
   v.init(m_ysize);
   for(int y=0; y<m_ysize; y++)
      v(y) = m_data[x][y];
   }

template <class T> inline void matrix<T>::insertcol(const vector<T>& v, const int y)
   {
   assert(v.size() == m_xsize);
   for(int x=0; x<m_xsize; x++)
      m_data[x][y] = v(x);
   }

template <class T> inline void matrix<T>::extractcol(vector<T>& v, const int y) const
   {
   v.init(m_xsize);
   for(int x=0; x<m_xsize; x++)
      v(x) = m_data[x][y];
   }

// convert to a vector

template <class T> inline matrix<T>::operator vector<T>() const
   {
   vector<T> v(size());
   int k=0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         v(k++) = m_data[i][j];
   return v;
   }

// index operators (perform boundary checking)

template <class T> inline T& matrix<T>::operator()(const int x, const int y)
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   return m_data[x][y];
   }

template <class T> inline T matrix<T>::operator()(const int x, const int y) const
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   return m_data[x][y];
   }

// serialization and stream input & output

template <class T> inline void matrix<T>::serialize(std::ostream& s) const
   {
   for(int j=0; j<m_ysize; j++)
      {
      s << m_data[0][j];
      for(int i=1; i<m_xsize; i++)
         s << "\t" << m_data[i][j];
      s << "\n";
      }
   }

template <class T> inline void matrix<T>::serialize(std::istream& s)
   {
   for(int j=0; j<m_ysize; j++)
      for(int i=0; i<m_xsize; i++)
         s >> m_data[i][j];
   }

template <class T> inline std::ostream& operator<<(std::ostream& s, const matrix<T>& x)
   {
   s << x.m_xsize << "\t" << x.m_ysize << "\n";
   x.serialize(s);
   return s;
   }

template <class T> inline std::istream& operator>>(std::istream& s, matrix<T>& x)
   {
   int xsize, ysize;
   s >> xsize >> ysize;
   x.setsize(xsize,ysize);
   x.serialize(s);
   return s;
   }

// comparison (mask-creation) operations

template <class T> inline matrix<bool> matrix<T>::operator==(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] == x.m_data[i][j]);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator!=(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] != x.m_data[i][j]);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator<=(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] <= x.m_data[i][j]);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator>=(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] >= x.m_data[i][j]);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator<(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] < x.m_data[i][j]);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator>(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] > x.m_data[i][j]);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator==(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] == x);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator!=(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] != x);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator<=(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] <= x);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator>=(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] >= x);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator<(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] < x);
   return r;
   }

template <class T> inline matrix<bool> matrix<T>::operator>(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] > x);
   return r;
   }

// arithmetic operations - unary

template <class T> inline matrix<T>& matrix<T>::operator+=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] += x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator-=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] -= x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator*=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] *= x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator/=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] /= x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator+=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] += x;
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator-=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] -= x;
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator*=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] *= x;
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator/=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] /= x;
   return *this;
   }

// arithmetic operations - binary

template <class T> inline matrix<T> matrix<T>::operator+(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r += x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator-(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r -= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator*(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r *= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator/(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r /= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator+(const T x) const
   {
   matrix<T> r = *this;
   r += x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator-(const T x) const
   {
   matrix<T> r = *this;
   r -= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator*(const T x) const
   {
   matrix<T> r = *this;
   r *= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator/(const T x) const
   {
   matrix<T> r = *this;
   r /= x;
   return r;
   }

// boolean operations - unary

template <class T> inline matrix<T>& matrix<T>::operator!()
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = !m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator&=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] &= x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator|=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] |= x.m_data[i][j];
   return *this;
   }

template <class T> inline matrix<T>& matrix<T>::operator^=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] ^= x.m_data[i][j];
   return *this;
   }

// boolean operations - binary

template <class T> inline matrix<T> matrix<T>::operator&(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r &= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator|(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r |= x;
   return r;
   }

template <class T> inline matrix<T> matrix<T>::operator^(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r ^= x;
   return r;
   }

// user-defined operations

template <class T> inline matrix<T>& matrix<T>::apply(T f(T))
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = f(m_data[i][j]);
   return *this;
   }

// statistical operations

template <class T> inline T matrix<T>::min() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0];
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         if(m_data[i][j] < result)
            result = m_data[i][j];
   return result;
   }

template <class T> inline T matrix<T>::max() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0];
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         if(m_data[i][j] > result)
            result = m_data[i][j];
   return result;
   }

template <class T> inline T matrix<T>::sum() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         result += m_data[i][j];
   return result;
   }

template <class T> inline T matrix<T>::sumsq() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         result += m_data[i][j] * m_data[i][j];
   return result;
   }

template <class T> inline T matrix<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

// *** masked matrix class ***

template <class T> class masked_matrix {
   friend class matrix<T>;
private:
   matrix<T>* m_data;
   matrix<bool> m_mask;
protected:
   masked_matrix(matrix<T>* data, const matrix<bool>& mask);
public:
   // matrix copy and value initialisation
   masked_matrix<T>& operator=(const T x);

   // convert to a vector
   operator vector<T>() const;

   // arithmetic operations - unary
   masked_matrix<T>& operator+=(const matrix<T>& x);
   masked_matrix<T>& operator-=(const matrix<T>& x);
   masked_matrix<T>& operator*=(const matrix<T>& x);
   masked_matrix<T>& operator/=(const matrix<T>& x);
   masked_matrix<T>& operator+=(const T x);
   masked_matrix<T>& operator-=(const T x);
   masked_matrix<T>& operator*=(const T x);
   masked_matrix<T>& operator/=(const T x);

   // user-defined operations
   masked_matrix<T>& apply(T f(T));

   // information services
   int size() const; // number of masked-in elements

   // statistical operations
   T min() const;
   T max() const;
   T sum() const;
   T sumsq() const;
   T mean() const { return sum()/T(size()); };
   T var() const;
   T sigma() const { return sqrt(var()); };
};

// masked matrix constructor

template <class T> inline masked_matrix<T>::masked_matrix(matrix<T>* data, const matrix<bool>& mask)
   {
   assert(data->m_xsize == mask.xsize());
   assert(data->m_ysize == mask.ysize());
   m_data = data;
   m_mask = mask;
   }

// matrix copy and value initialisation

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] = x;
   return *this;
   }

// convert to a vector

template <class T> inline masked_matrix<T>::operator vector<T>() const
   {
   vector<T> v(size());
   int k=0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            v(k++) = m_data->m_data[i][j];
   return v;
   }

// arithmetic operations - unary

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator+=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] += x.m_data[i][j];
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator-=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] -= x.m_data[i][j];
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator*=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] *= x.m_data[i][j];
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator/=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] /= x.m_data[i][j];
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator+=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] += x;
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator-=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] -= x;
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator*=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] *= x;
   return *this;
   }

template <class T> inline masked_matrix<T>& masked_matrix<T>::operator/=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] /= x;
   return *this;
   }

// user-defined operations

template <class T> inline masked_matrix<T>& masked_matrix<T>::apply(T f(T))
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] = f(m_data->m_data[i][j]);
   return *this;
   }

// information services

template <class T> inline int masked_matrix<T>::size() const
   {
   int result = 0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            result++;
   return result;
   }

// statistical operations

template <class T> inline T masked_matrix<T>::min() const
   {
   assert(size() > 0);
   T result;
   bool initial = true;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j) && (m_data->m_data[i][j] < result || initial))
            {
            result = m_data->m_data[i][j];
            initial = false;
            }
   return result;
   }

template <class T> inline T masked_matrix<T>::max() const
   {
   assert(size() > 0);
   T result;
   bool initial = true;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j) && (m_data->m_data[i][j] > result || initial))
            {
            result = m_data->m_data[i][j];
            initial = false;
            }
   return result;
   }

template <class T> inline T masked_matrix<T>::sum() const
   {
   assert(m_data->m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            result += m_data->m_data[i][j];
   return result;
   }

template <class T> inline T masked_matrix<T>::sumsq() const
   {
   assert(m_data->m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            result += m_data->m_data[i][j] * m_data->m_data[i][j];
   return result;
   }

template <class T> inline T masked_matrix<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

}; // end namespace

#endif
