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

#ifndef __matrix3_h
#define __matrix3_h

#include "config.h"
#include <cstdlib>
#include <iostream>

namespace libbase {

/*!
 * \brief   Generic 3D Matrix.
 * \author  Johann Briffa
 *
 * \version 1.00 (31 Oct 2001)
 * separated 3D matrix class from the 2D matrix header.
 * Created arithmetic functions as part of the matrix class. These are only created
 * for an instantiation in which they are used, so it should not pose a problem anyway.
 * This includes arithmetic operations between matrices, constant matrix initialisation
 * routines, and also some statistical functions. Also, cleaned up the protected
 * (internal) interface, and formalised the existence of empty matrices. Also renamed
 * member variables to start with m_ in order to facilitate the similar naming of public
 * functions (such as the size functions).
 *
 * \version 1.21 (2 Dec 2001)
 * added a function which sets the size of a matrix to the given size - leaving it as
 * it is if the size was already good, and freeing/reallocating if necessary. This helps
 * reduce redundant free/alloc operations on matrices which keep the same size.
 * [Ported from matrix 1.11]
 *
 * \version 1.50 (13 Apr 2002)
 * added a number of high-level support routines for working with matrices - the overall
 * effect of this should be a drastic reduction in the number of loops required in user
 * code to express various common operations. Changes are:
 * - support for working with different-sized matrices (in place of resizing operations
 * which would be quite expensive); added a function copyfrom() which copies data from
 * another matrix without resizing this one. Opted for this rather than changing the
 * definition of operator= because it's convenient for '=' to copy _everything_ from the
 * source to the destination; otherwise we would land into obscure problems in some cases
 * (like when we're trying to copy a vector of matrices, etc.). This method also has the
 * advantage of keeping the old code/interface as it was.
 * - added a new format for init(), which takes another matrix as argument, to allow
 * easier (and neater) sizing of one matrix based on another. This is a template function
 * to allow the argument matrix to be of a different type.
 * - added an apply() function which allows the user to do the same operation on all
 * elements (previously had to do this manually).
 *
 * \version 1.60 (9 May 2002)
 * - added another apply() so that the given function's parameter is const - this allows
 * the use of functions which do not modify their parameter (it's actually what we want
 * anyway). The older version (with non-const parameter) is still kept to allow the use
 * of functions where the parameter is not defined as const (such as fabs, etc).
 * - added unary and binary boolean operators.
 * - also, changed the binary operators to be member functions with a single argument,
 * rather than non-members with two arguments. Also, for operations with a constant
 * (rather than another vector), that constant is passed directly, not by reference.
 * - added serialize() functions which read and write vector data only; the input function
 * assumes that the current vector already has the correct size. These functions are
 * useful for interfacing with other file formats. Also modified the stream I/O functions
 * to make use of these.
 *
 *
 * \todo Consider removing this class, and port any existing uses to boost
 * multi_array
 */

template <class T>
class matrix3;

template <class T>
std::ostream& operator<<(std::ostream& s, const matrix3<T>& x);
template <class T>
std::istream& operator>>(std::istream& s, matrix3<T>& x);

template <class T>
class matrix3 {
private:
   int m_xsize, m_ysize, m_zsize;
   T ***m_data;
protected:
   // memory allocation functions
   void alloc(const int x, const int y, const int z); // allocates memory for (x,y,z) elements and updates sizes
   void free(); // if there is memory allocated, free it
   void setsize(const int x, const int y, const int z); // set matrix to given size, freeing if and as required
public:
   matrix3(const int x = 0, const int y = 0, const int z = 0); // constructor (does not initialise elements)
   matrix3(const matrix3<T>& x); // copy constructor
   ~matrix3();

   // resizing operations
   void init(const int x, const int y, const int z);
   template <class A>
   void init(const matrix3<A>& x)
      {
      init(x.m_xsize, x.m_ysize, x.m_zsize);
      }

   // matrix3 copy and value initialisation
   matrix3<T>& copyfrom(const matrix3<T>& x);
   matrix3<T>& operator=(const matrix3<T>& x);
   matrix3<T>& operator=(const T x);

   // index operators (perform boundary checking)
   T& operator()(const int x, const int y, const int z);
   const T& operator()(const int x, const int y, const int z) const;

   // information services
   //! Total number of elements
   int size() const
      {
      return m_xsize * m_ysize * m_zsize;
      }
   int xsize() const
      {
      return m_xsize;
      }
   int ysize() const
      {
      return m_ysize;
      }
   int zsize() const
      {
      return m_zsize;
      }

   // serialization and stream input & output
   void serialize(std::ostream& s) const;
   void serialize(std::istream& s);

   // arithmetic operations - unary
   matrix3<T>& operator+=(const matrix3<T>& x);
   matrix3<T>& operator-=(const matrix3<T>& x);
private:
   matrix3<T>& operator*=(const matrix3<T>& x);
   matrix3<T>& operator/=(const matrix3<T>& x);
public:
   matrix3<T>& operator+=(const T x);
   matrix3<T>& operator-=(const T x);
   matrix3<T>& operator*=(const T x);
   matrix3<T>& operator/=(const T x);

   // arithmetic operations - binary
   matrix3<T> operator+(const matrix3<T>& x) const;
   matrix3<T> operator-(const matrix3<T>& x) const;
private:
   matrix3<T> operator*(const matrix3<T>& x) const;
   matrix3<T> operator/(const matrix3<T>& x) const;
public:
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

// memory allocation functions

template <class T>
inline void matrix3<T>::alloc(const int x, const int y, const int z)
   {
   if (x == 0 || y == 0 || z == 0)
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
      for (int i = 0; i < x; i++)
         {
         m_data[i] = new Tp[y];
         for (int j = 0; j < y; j++)
            m_data[i][j] = new T[z];
         }
      }
   }

template <class T>
inline void matrix3<T>::free()
   {
   // note that if xsize is 0, then ysize & zsize must also be zero
   if (m_xsize > 0)
      {
      for (int i = 0; i < m_xsize; i++)
         {
         for (int j = 0; j < m_ysize; j++)
            delete[] m_data[i][j];
         delete[] m_data[i];
         }
      delete[] m_data;
      }
   }

template <class T>
inline void matrix3<T>::setsize(const int x, const int y, const int z)
   {
   if (x == m_xsize && y == m_ysize && z == m_zsize)
      return;
   free();
   alloc(x, y, z);
   }

// constructor / destructor functions

template <class T>
inline matrix3<T>::matrix3(const int x, const int y, const int z)
   {
   assert(x >= 0);
   assert(y >= 0);
   assert(z >= 0);
   alloc(x, y, z);
   }

template <class T>
inline matrix3<T>::matrix3(const matrix3<T>& x)
   {
   alloc(x.m_xsize, x.m_ysize, x.m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] = x.m_data[i][j][k];
   }

template <class T>
inline matrix3<T>::~matrix3()
   {
   free();
   }

// resizing operations

template <class T>
inline void matrix3<T>::init(const int x, const int y, const int z)
   {
   assert(x >= 0);
   assert(y >= 0);
   assert(z >= 0);
   setsize(x, y, z);
   }

// matrix copy and value initialisation

template <class T>
inline matrix3<T>& matrix3<T>::copyfrom(const matrix3<T>& x)
   {
   const int xsize = std::min(m_xsize, x.m_xsize);
   const int ysize = std::min(m_ysize, x.m_ysize);
   const int zsize = std::min(m_zsize, x.m_zsize);
   for (int i = 0; i < xsize; i++)
      for (int j = 0; j < ysize; j++)
         for (int k = 0; k < zsize; k++)
            m_data[i][j][k] = x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator=(const matrix3<T>& x)
   {
   setsize(x.m_xsize, x.m_ysize, x.m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] = x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator=(const T x)
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] = x;
   return *this;
   }

// index operators (perform boundary checking)

template <class T>
inline T& matrix3<T>::operator()(const int x, const int y, const int z)
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   assert(z>=0 && z<m_zsize);
   return m_data[x][y][z];
   }

template <class T>
inline const T& matrix3<T>::operator()(const int x, const int y, const int z) const
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   assert(z>=0 && z<m_zsize);
   return m_data[x][y][z];
   }

// serialization and stream input & output

template <class T>
inline void matrix3<T>::serialize(std::ostream& s) const
   {
   for (int k = 0; k < m_zsize; k++)
      {
      s << std::endl;
      for (int j = 0; j < m_ysize; j++)
         {
         s << m_data[0][j][k];
         for (int i = 1; i < m_xsize; i++)
            s << "\t" << m_data[i][j][k];
         s << std::endl;
         }
      }
   }

template <class T>
inline void matrix3<T>::serialize(std::istream& s)
   {
   for (int k = 0; k < m_zsize; k++)
      for (int j = 0; j < m_ysize; j++)
         for (int i = 0; i < m_xsize; i++)
            s >> m_data[i][j][k];
   }

template <class T>
inline std::ostream& operator<<(std::ostream& s, const matrix3<T>& x)
   {
   s << x.xsize() << "\t" << x.ysize() << "\t" << x.zsize() << std::endl;
   x.serialize(s);
   return s;
   }

template <class T>
inline std::istream& operator>>(std::istream& s, matrix3<T>& x)
   {
   int xsize, ysize, zsize;
   s >> xsize >> ysize >> zsize;
   x.init(xsize, ysize, zsize);
   x.serialize(s);
   return s;
   }

// arithmetic operations - unary

template <class T>
inline matrix3<T>& matrix3<T>::operator+=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] += x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator-=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] -= x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator*=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] *= x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator/=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] /= x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator+=(const T x)
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] += x;
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator-=(const T x)
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] -= x;
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator*=(const T x)
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] *= x;
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator/=(const T x)
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] /= x;
   return *this;
   }

// arithmetic operations - binary

template <class T>
inline matrix3<T> matrix3<T>::operator+(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r += x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator-(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r -= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator*(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r *= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator/(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r /= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator+(const T x) const
   {
   matrix3<T> r = *this;
   r += x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator-(const T x) const
   {
   matrix3<T> r = *this;
   r -= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator*(const T x) const
   {
   matrix3<T> r = *this;
   r *= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator/(const T x) const
   {
   matrix3<T> r = *this;
   r /= x;
   return r;
   }

// boolean operations - unary

template <class T>
inline matrix3<T>& matrix3<T>::operator!()
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] = !m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator&=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] &= x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator|=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] |= x.m_data[i][j][k];
   return *this;
   }

template <class T>
inline matrix3<T>& matrix3<T>::operator^=(const matrix3<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   assert(x.m_zsize == m_zsize);
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] ^= x.m_data[i][j][k];
   return *this;
   }

// boolean operations - binary

template <class T>
inline matrix3<T> matrix3<T>::operator&(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r &= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator|(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r |= x;
   return r;
   }

template <class T>
inline matrix3<T> matrix3<T>::operator^(const matrix3<T>& x) const
   {
   matrix3<T> r = *this;
   r ^= x;
   return r;
   }

// user-defined operations

template <class T>
inline matrix3<T>& matrix3<T>::apply(T f(T))
   {
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            m_data[i][j][k] = f(m_data[i][j][k]);
   return *this;
   }

// statistical operations

template <class T>
inline T matrix3<T>::min() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0][0];
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            if (m_data[i][j][k] < result)
               result = m_data[i][j][k];
   return result;
   }

template <class T>
inline T matrix3<T>::max() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0][0];
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            if (m_data[i][j][k] > result)
               result = m_data[i][j][k];
   return result;
   }

template <class T>
inline T matrix3<T>::sum() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            result += m_data[i][j][k];
   return result;
   }

template <class T>
inline T matrix3<T>::sumsq() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for (int i = 0; i < m_xsize; i++)
      for (int j = 0; j < m_ysize; j++)
         for (int k = 0; k < m_zsize; k++)
            result += m_data[i][j][k] * m_data[i][j][k];
   return result;
   }

template <class T>
inline T matrix3<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq() / T(size()) - _mean * _mean;
   return (_var > 0) ? _var : 0;
   }

} // end namespace

#endif
