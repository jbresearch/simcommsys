#ifndef __matrix_h
#define __matrix_h

#include "config.h"
#include "vector.h"
#include <stdlib.h>
#include <iostream>
#include <algorithm>

namespace libbase {

/* \note
   To comply with the standard, matrix's friend functions must be declared
   before the main class. Consequently, a declaration of the class itself
   is also required before that.
*/
template <class T>
class matrix;
template <class T>
class masked_matrix;

template <class T>
std::ostream& operator<<(std::ostream& s, const matrix<T>& x);
template <class T>
std::istream& operator>>(std::istream& s, matrix<T>& x);

/*!
   \brief   Generic 2D Matrix.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Arithmetic functions are part of the matrix class. This includes arithmetic
   operations between matrices, constant matrix initialisation routines, and
   some statistical functions.

   \note Empty matrices (that is, ones with no elements) are defined and valid.

   \note Range-checking and other validation functions are only operative in
         debug mode.

   \todo Change the convention for row/column in the class (note this will
         require changes wherever this class is used!). This class needs to be
         re-designed in a manner that is consistent with convention (esp.
         Matlab) and that is efficient.
*/

template <class T>
class matrix {
   friend class masked_matrix<T>;
private:
   int  m_xsize, m_ysize;
   T    **m_data;
protected:
   // memory allocation functions
   void alloc(const int x, const int y);
   void free();
   void setsize(const int x, const int y);
public:
   /*! \brief Default constructor
      This exists instead of default-values in principal constructor to avoid
      allowing the situation where a constructor is given a _single_ integer
      parameter (i.e. x has a value, y takes its default)
   */
   matrix() { alloc(0,0); };
   /*! \brief Principal constructor
      \note Does not initialize elements.
   */
   matrix(const int x, const int y) { alloc(x,y); };
   matrix(const matrix<T>& m);
   ~matrix() { free(); };

   // resizing operations
   //! \copydoc setsize()
   void init(const int x, const int y) { setsize(x,y); };
   /*! \copydoc setsize()
      This overload takes another matrix as argument, to allow easier (and
      neater) sizing of one matrix based on another. This is a template
      function to allow the argument matrix to be of a different type.
   */
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
   int xsize() const { return m_xsize; };                //!< Size on dimension x (rows)
   int ysize() const { return m_ysize; };                //!< Size on dimension y (columns)
   int size() const { return m_xsize * m_ysize; };       //!< Total number of elements

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

   // direct comparison operations
   bool isequalto(const matrix<T>& x) const;
   bool isnotequalto(const matrix<T>& x) const;

   // arithmetic operations - unary
   matrix<T>& operator+=(const matrix<T>& x);
   matrix<T>& operator-=(const matrix<T>& x);
   matrix<T>& operator*=(const matrix<T>& x);
   matrix<T>& operator/=(const matrix<T>& x);
   matrix<T>& multiplyby(const matrix<T>& x);
   matrix<T>& divideby(const matrix<T>& x);
   matrix<T>& operator+=(const T x);
   matrix<T>& operator-=(const T x);
   matrix<T>& operator*=(const T x);
   matrix<T>& operator/=(const T x);

   // arithmetic operations - binary
   matrix<T> operator+(const matrix<T>& x) const;
   matrix<T> operator-(const matrix<T>& x) const;
   matrix<T> operator*(const matrix<T>& x) const;
   vector<T> operator*(const vector<T>& x) const;
   matrix<T> operator/(const matrix<T>& x) const;
   matrix<T> multiply(const matrix<T>& x) const;
   matrix<T> divide(const matrix<T>& x) const;
   matrix<T> operator+(const T x) const;
   matrix<T> operator-(const T x) const;
   matrix<T> operator*(const T x) const;
   matrix<T> operator/(const T x) const;

   // boolean operations - modifying
   matrix<T>& operator&=(const matrix<T>& x);
   matrix<T>& operator|=(const matrix<T>& x);
   matrix<T>& operator^=(const matrix<T>& x);

   // boolean operations - non-modifying
   matrix<T> operator!() const;
   matrix<T> operator&(const matrix<T>& x) const;
   matrix<T> operator|(const matrix<T>& x) const;
   matrix<T> operator^(const matrix<T>& x) const;

   // user-defined operations
   matrix<T>& apply(T f(T));

   // matrix-arithmetic operations
   matrix<T> inverse() const;
   matrix<T> transpose() const;

   // statistical operations
   T min() const;
   T max() const;
   T sum() const;
   T sumsq() const;
   //! Compute the mean value of matrix elements
   T mean() const { return sum()/T(size()); };
   T var() const;
   //! Compute standard deviation of matrix elements
   T sigma() const { return sqrt(var()); };

   // static functions
   static matrix<T> eye(int n);
};


//! If there is memory allocated, free it
template <class T>
inline void matrix<T>::free()
   {
   // note that if xsize is 0, then ysize must also be zero
   if(m_xsize > 0)
      {
      for(int i=0; i<m_xsize; i++)
         delete[] m_data[i];
      delete[] m_data;
      }
   }

/*! \brief Set matrix to given size, freeing if and as required
   
   This method leaves the matrix as it is if the size was already correct, and
   frees/reallocates if necessary. This helps reduce redundant free/alloc
   operations on matrices which keep the same size.
*/
template <class T>
inline void matrix<T>::setsize(const int x, const int y)
   {
   if(x==m_xsize && y==m_ysize)
      return;
   free();
   alloc(x,y);
   }

/*! \brief Allocates memory for (x,y) elements and updates sizes
   \note Detects invalid size values (either x or y being 0, but not both)
*/
template <class T>
inline void matrix<T>::alloc(const int x, const int y)
   {
   if(x==0 && y==0)
      {
      m_xsize = 0;
      m_ysize = 0;
      m_data = NULL;
      }
   else
      {
      assertalways(x>0 && y>0);
      m_xsize = x;
      m_ysize = y;
      typedef T* Tp;
      m_data = new Tp[x];
      for(int i=0; i<x; i++)
         m_data[i] = new T[y];
      }
   }

// constructor / destructor functions

template <class T>
inline matrix<T>::matrix(const matrix<T>& m)
   {
   alloc(m.m_xsize, m.m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = m.m_data[i][j];
   }

// matrix copy and value initialisation

/*! \brief Copies data from another matrix without resizing this one
   
   Adds support for working with different-sized matrices (in place of
   resizing operations which would be quite expensive).
   
   \note Opted for this rather than changing the definition of operator=
         because it's convenient for '=' to copy _everything_ from the source
         to the destination; otherwise we would land into obscure problems in
         some cases (like when we're trying to copy a vector of matrices).
*/
template <class T>
inline matrix<T>& matrix<T>::copyfrom(const matrix<T>& x)
   {
   const int xsize = std::min(m_xsize, x.m_xsize);
   const int ysize = std::min(m_ysize, x.m_ysize);
   for(int i=0; i<xsize; i++)
      for(int j=0; j<ysize; j++)
         m_data[i][j] = x.m_data[i][j];
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator=(const matrix<T>& x)
   {
   setsize(x.m_xsize, x.m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = x.m_data[i][j];
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = x;
   return *this;
   }

// insert/extract rows/columns as vectors

/*! \brief Insert vector into row 'x'
*/
template <class T>
inline void matrix<T>::insertrow(const vector<T>& v, const int x)
   {
   assert(v.size() == m_ysize);
   for(int y=0; y<m_ysize; y++)
      m_data[x][y] = v(y);
   }

/*! \brief Extract row 'x' as a vector
   The target vector needs to be passed as a parameter; the expression format
   can be improved aesthetically, however the present format clearly
   communicates what is happening.
*/
template <class T>
inline void matrix<T>::extractrow(vector<T>& v, const int x) const
   {
   v.init(m_ysize);
   for(int y=0; y<m_ysize; y++)
      v(y) = m_data[x][y];
   }

/*! \brief Insert vector into column 'y'
*/
template <class T>
inline void matrix<T>::insertcol(const vector<T>& v, const int y)
   {
   assert(v.size() == m_xsize);
   for(int x=0; x<m_xsize; x++)
      m_data[x][y] = v(x);
   }

/*! \brief Extract column 'y' as a vector
   The target vector needs to be passed as a parameter; the expression format
   can be improved aesthetically, however the present format clearly
   communicates what is happening.
*/
template <class T>
inline void matrix<T>::extractcol(vector<T>& v, const int y) const
   {
   v.init(m_xsize);
   for(int x=0; x<m_xsize; x++)
      v(x) = m_data[x][y];
   }

/*! \brief Convert matrix to a vector
   Elements are extracted in column-major order (ie. starting with the top-left
   first go down then across).
*/
template <class T>
inline matrix<T>::operator vector<T>() const
   {
   vector<T> v(size());
   int k=0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         v(k++) = m_data[i][j];
   return v;
   }

// index operators (perform boundary checking)

template <class T>
inline T& matrix<T>::operator()(const int x, const int y)
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   return m_data[x][y];
   }

template <class T>
inline T matrix<T>::operator()(const int x, const int y) const
   {
   assert(x>=0 && x<m_xsize);
   assert(y>=0 && y<m_ysize);
   return m_data[x][y];
   }

/*! \brief Writes matrix data to output stream.
   This function is intended for interfacing with file formats that do not use
   the serialization format of this class.
*/
template <class T>
inline void matrix<T>::serialize(std::ostream& s) const
   {
   for(int j=0; j<m_ysize; j++)
      {
      s << m_data[0][j];
      for(int i=1; i<m_xsize; i++)
         s << "\t" << m_data[i][j];
      s << "\n";
      }
   }

/*! \brief Reads matrix data from input stream.
   \note Assumes that the current matrix already has the correct size.
   This function is intended for interfacing with file formats that do not use
   the serialization format of this class.
*/
template <class T>
inline void matrix<T>::serialize(std::istream& s)
   {
   for(int j=0; j<m_ysize; j++)
      for(int i=0; i<m_xsize; i++)
         s >> m_data[i][j];
   }

/*! \brief Writes matrix to output stream.
   Includes matrix size, to allow correct reconstruction when reading in.
*/
template <class T>
inline std::ostream& operator<<(std::ostream& s, const matrix<T>& x)
   {
   s << x.m_xsize << "\t" << x.m_ysize << "\n";
   x.serialize(s);
   return s;
   }

/*! \brief Reads and reconstructs matrix from input stream.
*/
template <class T>
inline std::istream& operator>>(std::istream& s, matrix<T>& x)
   {
   int xsize, ysize;
   s >> xsize >> ysize;
   x.setsize(xsize,ysize);
   x.serialize(s);
   return s;
   }

// comparison (mask-creation) operations

template <class T>
inline matrix<bool> matrix<T>::operator==(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] == x.m_data[i][j]);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator!=(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] != x.m_data[i][j]);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator<=(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] <= x.m_data[i][j]);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator>=(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] >= x.m_data[i][j]);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator<(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] < x.m_data[i][j]);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator>(const matrix<T>& x) const
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] > x.m_data[i][j]);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator==(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] == x);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator!=(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] != x);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator<=(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] <= x);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator>=(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] >= x);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator<(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] < x);
   return r;
   }

template <class T>
inline matrix<bool> matrix<T>::operator>(const T x) const
   {
   matrix<bool> r(m_xsize, m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r(i,j) = (m_data[i][j] > x);
   return r;
   }

// comparison (mask-creation) operations

template <class T>
inline bool matrix<T>::isequalto(const matrix<T>& x) const
   {
   if(x.m_xsize != m_xsize)
      return false;
   if(x.m_ysize != m_ysize)
      return false;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         if(m_data[i][j] != x.m_data[i][j])
            return false;
   return true;
   }

template <class T>
inline bool matrix<T>::isnotequalto(const matrix<T>& x) const
   {
   return !isequalto(x);
   }

// arithmetic operations - unary

template <class T>
inline matrix<T>& matrix<T>::operator+=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] += x.m_data[i][j];
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator-=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] -= x.m_data[i][j];
   return *this;
   }

/*!
   \brief Ordinary matrix multiplication
   \param  x   Matrix to be multiplied to this one
   \return The updated (multiplied-into) matrix
*/
template <class T>
inline matrix<T>& matrix<T>::operator*=(const matrix<T>& x)
   {
   matrix<T> r = *this * x;
   return *this = r;
   }

/*!
   \brief Ordinary matrix division
   \param  x   Matrix to divide this one by
   \return The updated (divided-by) matrix
*/
template <class T>
inline matrix<T>& matrix<T>::operator/=(const matrix<T>& x)
   {
   matrix<T> r = *this / x;
   return *this = r;
   }

/*!
   \brief Array multiplication (element-by-element) of matrices
   \param  x   Matrix to be multiplied to this one
   \return The updated (multiplied-into) matrix
*/
template <class T>
inline matrix<T>& matrix<T>::multiplyby(const matrix<T>& x)
   {
   // for A.*B:
   // The size of A must be the same as the size of B.
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] *= x.m_data[i][j];
   return *this;
   }

/*!
   \brief Array division (element-by-element) of matrices
   \param  x   Matrix to divide this one by
   \return The updated (divided-into) matrix
*/
template <class T>
inline matrix<T>& matrix<T>::divideby(const matrix<T>& x)
   {
   // for A./B:
   // The size of A must be the same as the size of B.
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] /= x.m_data[i][j];
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator+=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] += x;
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator-=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] -= x;
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator*=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] *= x;
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator/=(const T x)
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] /= x;
   return *this;
   }

// arithmetic operations - binary

template <class T>
inline matrix<T> matrix<T>::operator+(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r += x;
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator-(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r -= x;
   return r;
   }

/*!
   \brief Ordinary matrix multiplication
   \param  x   Matrix to be multiplied to this one
   \return The result of 'this' multiplied by 'x'
   \note The use of 'i' and 'j' indices in this function follows the mathematical convention,
         rather than that used in the rest of this class.
*/
template <class T>
inline matrix<T> matrix<T>::operator*(const matrix<T>& x) const
   {
   // for A.B:
   // The number of columns of A must be the same as the number of rows of B.
   assert(m_xsize == x.m_ysize);
   // If A is an m-by-n matrix and B is an n-by-p matrix, then the product is an m-by-p matrix 
   matrix<T> r(x.m_xsize,m_ysize);
   // Element AB_{i,j} = \sum_{k=1}^{n} a_{i,k} b_{k,j}
   for(int i=0; i<r.m_ysize; i++)
      for(int j=0; j<r.m_xsize; j++)
         {
         r.m_data[j][i] = 0;
         for(int k=0; k<m_xsize; k++)
            r.m_data[j][i] += m_data[k][i] * x.m_data[j][k];
         }
   return r;
   }

/*!
   \brief Ordinary matrix multiplication by column vector
   \param  x   Vector to be multiplied to this matrix
   \return The result of 'this' multiplied by 'x'
   \warning The use of 'i' and 'j' indices in this function follows the
            mathematical convention, rather than that used in the rest of this
            class.
*/
template <class T>
inline vector<T> matrix<T>::operator*(const vector<T>& x) const
   {
   // for A.B:
   // The number of columns of A must be the same as the number of rows of B.
   assert(m_xsize == x.size());
   // If A is an m-by-n matrix and B is an n-by-1 vector, then the product is an m-by-1 matrix 
   vector<T> r(m_ysize);
   // Element AB_{i} = \sum_{k=1}^{n} a_{i,k} b_{k}
   for(int i=0; i<r.size(); i++)
      {
      r(i) = 0;
      for(int k=0; k<m_xsize; k++)
         r(i) += m_data[k][i] * x(k);
      }
   return r;
   }

/*!
   \brief Ordinary matrix division by matrix inversion
   \param  x   Matrix to divide this one by
   \return The result of 'this' divided by 'x'
*/
template <class T>
inline matrix<T> matrix<T>::operator/(const matrix<T>& x) const
   {
   return *this * x.inverse();
   }

/*!
   \brief Array multiplication (element-by-element) of matrices
   \param  x   Matrix to be multiplied to this one
   \return The result of 'this' multiplied by 'x'
*/
template <class T>
inline matrix<T> matrix<T>::multiply(const matrix<T>& x) const
   {
   // for A.*B:
   // The size of A must be the same as the size of B.
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   // Result is same size
   matrix<T> r(m_xsize,m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r = m_data[i][j] * x.m_data[i][j];
   return r;
   }

/*!
   \brief Array division (element-by-element) of matrices
   \param  x   Matrix to divide this one by
   \return The result of 'this' divided by 'x'
*/
template <class T>
inline matrix<T> matrix<T>::divide(const matrix<T>& x) const
   {
   // for A./B:
   // The size of A must be the same as the size of B.
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   // Result is same size
   matrix<T> r(m_xsize,m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r = m_data[i][j] / x.m_data[i][j];
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator+(const T x) const
   {
   matrix<T> r = *this;
   r += x;
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator-(const T x) const
   {
   matrix<T> r = *this;
   r -= x;
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator*(const T x) const
   {
   matrix<T> r = *this;
   r *= x;
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator/(const T x) const
   {
   matrix<T> r = *this;
   r /= x;
   return r;
   }

// boolean operations - modifying

template <class T>
inline matrix<T>& matrix<T>::operator&=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] &= x.m_data[i][j];
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator|=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] |= x.m_data[i][j];
   return *this;
   }

template <class T>
inline matrix<T>& matrix<T>::operator^=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_xsize);
   assert(x.m_ysize == m_ysize);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] ^= x.m_data[i][j];
   return *this;
   }

// boolean operations - non-modifying

template <class T>
inline matrix<T> matrix<T>::operator!() const
   {
   matrix<T> r(*this);
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r.m_data[i][j] = !m_data[i][j];
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator&(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r &= x;
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator|(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r |= x;
   return r;
   }

template <class T>
inline matrix<T> matrix<T>::operator^(const matrix<T>& x) const
   {
   matrix<T> r = *this;
   r ^= x;
   return r;
   }

/*! \brief Perform user-defined operation on all matrix elements

   \note Removed the instance of apply() whose given function's parameter is
         const, since this was causing problems with gcc on Solaris.
*/
template <class T>
inline matrix<T>& matrix<T>::apply(T f(T))
   {
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         m_data[i][j] = f(m_data[i][j]);
   return *this;
   }

// matrix-arithmetic operations

/*!
   \brief Matrix inversion, by direct Gauss-Jordan elimination
   \return The inverse of this matrix
   \invariant Matrix must be square
   \note Template class must provide the subtraction, division, and multiplication
   \note Template class must provide conversion to/from integer
   Performs row pivoting when necessary.
*/
template <class T>
inline matrix<T> matrix<T>::inverse() const
   {
   assertalways(m_xsize == m_ysize);
   const int n = m_xsize;
   matrix<T> r = eye(n);
   // create copy of rows of this matrix
   vector< vector<T> > arows(n);
   for(int i=0; i<n; i++)
      extractrow(arows(i), i);
   // create copy of rows of identity
   vector< vector<T> > rrows(n);
   for(int i=0; i<n; i++)
      r.extractrow(rrows(i), i);
   // perform Gauss-Jordan elimination
   // repeat for all rows
   for(int i=0; i<n; i++)
      {
      //trace << "DEBUG (matrix): G-J elimination on row " << i << "\n";
      //trace << "DEBUG (matrix): A = " << arows;
      //trace << "DEBUG (matrix): R = " << rrows;
      // find a suitable pivot element
      if(arows(i)(i) == 0)
         for(int j=i+1; j<n; j++)
            if(arows(j)(i) != 0)
               {
               std::swap(rrows(i),rrows(j));
               std::swap(arows(i),arows(j));
               //trace << "DEBUG (matrix): swapped rows " << i << "<->" << j << "\n";
               //trace << "DEBUG (matrix): A = " << arows;
               //trace << "DEBUG (matrix): R = " << rrows;
               break;
               }
      // divide by pivot element
      rrows(i) /= arows(i)(i);
      arows(i) /= arows(i)(i);
      // subtract the required amount from all other rows
      for(int j=0; j<n; j++)
         {
         if(j == i)
            continue;
         rrows(j) -= rrows(i) * arows(j)(i);
         arows(j) -= arows(i) * arows(j)(i);
         }
      }
   // copy back rows of result
   for(int i=0; i<n; i++)
      r.insertrow(rrows(i), i);
   return r;
   }

/*!
   \brief Matrix transpose
*/
template <class T>
inline matrix<T> matrix<T>::transpose() const
   {
   matrix<T> r(m_ysize,m_xsize);
   // copy over data
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         r.m_data[j][i] = m_data[i][j];
   return r;
   }

// statistical operations

/*! \brief Determines the smallest element in the matrix

   \note This assumes that less-than comparison is defined in operator (<).
   \note This is only valid for non-empty matrices.
*/
template <class T>
inline T matrix<T>::min() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0];
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         if(m_data[i][j] < result)
            result = m_data[i][j];
   return result;
   }

/*! \brief Determines the largest element in the matrix

   \note This assumes that greater-than comparison is defined in operator (>).
   \note This is only valid for non-empty matrices.
*/
template <class T>
inline T matrix<T>::max() const
   {
   assert(m_xsize > 0);
   T result = m_data[0][0];
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         if(m_data[i][j] > result)
            result = m_data[i][j];
   return result;
   }

/*! \brief Computes sum of elements in matrix

   \note This assumes that addition is defined for the type, in the accumulate
         operator (+=). Also, it is assumed that '0' is defined for the type.
*/
template <class T>
inline T matrix<T>::sum() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         result += m_data[i][j];
   return result;
   }

/*! \brief Computes sum of squares of elements in matrix

   \note This assumes that addition is defined for the type, in the accumulate
         operator (+=), as well as multiplication in the binary operator (*).
         Also, it is assumed that '0' is defined for the type.
*/
template <class T>
inline T matrix<T>::sumsq() const
   {
   assert(m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_xsize; i++)
      for(int j=0; j<m_ysize; j++)
         result += m_data[i][j] * m_data[i][j];
   return result;
   }

/*! \brief Computes the variance of elements
*/
template <class T>
inline T matrix<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

// static functions

/*!
   \brief Identity matrix
   \return The 'n'x'n' identity matrix
*/
template <class T>
inline matrix<T> matrix<T>::eye(int n)
   {
   assert(n > 0);
   matrix<T> r(n,n);
   r = 0;
   for(int i=0; i<n; i++)
      r(i,i) = 1;
   return r;
   }

}; // end namespace

/*!
   \brief Ordinary matrix power
   \return The result of A^n using ordinary matrix multiplication
   \note The use of 'i' and 'j' indices in this function follows the mathematical convention,
         rather than that used in the rest of this class.
*/
template <class T>
inline libbase::matrix<T> pow(const libbase::matrix<T>& A, int n)
   {
   using libbase::matrix;
   assert(A.xsize() == A.ysize());
   // power by zero return identity
   if(n == 0)
      return matrix<T>::eye(A.xsize());
   // handle negative powers as powers of the inverse
   matrix<T> R;
   if(n > 0)
      R = A;
   else
      {
      R = A.inverse();
      n = -n;
      }
   // square for as long as possible
   for(; n>0 && n%2==0; n/=2)
      R *= R;
   // repeatedly multiply by A for whatever remains
   for(n--; n>0; n--)
      R *= A;
   return R;
   }


namespace libbase {

/*!
   \brief   Masked 2D Matrix.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   A masked matrix is a matrix with a binary element-mask. Arithmetic,
   statistical, user-defined operation, and copy/value init functions are
   defined for this class, allowing us to modify the masked parts of any
   given matrix with ease.
    
   It is intended that for the user, the use of masked matrices should be
   essentially transparent (in that they can mostly be used in place of
   normal matrices) and that the user should never create one explicitly,
   but merely through a normal matrix.
*/
template <class T>
class masked_matrix {
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
private:
   //! Matrix multiplication is hidden because it's meaningless here
   masked_matrix<T>& operator*=(const matrix<T>& x) {};
   //! Matrix division is hidden because it's meaningless here
   masked_matrix<T>& operator/=(const matrix<T>& x) {};
public:
   masked_matrix<T>& multiplyby(const matrix<T>& x);
   masked_matrix<T>& divideby(const matrix<T>& x);
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

template <class T>
inline masked_matrix<T>::masked_matrix(matrix<T>* data, const matrix<bool>& mask)
   {
   assert(data->m_xsize == mask.xsize());
   assert(data->m_ysize == mask.ysize());
   m_data = data;
   m_mask = mask;
   }

// matrix copy and value initialisation

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] = x;
   return *this;
   }

// convert to a vector

template <class T>
inline masked_matrix<T>::operator vector<T>() const
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

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator+=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] += x.m_data[i][j];
   return *this;
   }

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator-=(const matrix<T>& x)
   {
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] -= x.m_data[i][j];
   return *this;
   }

/*!
   \brief Array multiplication (element-by-element) of matrices
   \param  x   Matrix to be multiplied to this one
   \return The updated (multiplied-into) matrix

   Masked elements (ie. where the mask is true) are multiplied by
   the corresponding element in 'x'. Unmasked elements are left
   untouched.
*/
template <class T>
inline masked_matrix<T>& masked_matrix<T>::multiplyby(const matrix<T>& x)
   {
   // for A.*B:
   // The size of A must be the same as the size of B.
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] *= x.m_data[i][j];
   return *this;
   }

/*!
   \brief Array division (element-by-element) of matrices
   \param  x   Matrix to divide this one by
   \return The updated (divided-into) matrix

   Masked elements (ie. where the mask is true) are divided by
   the corresponding element in 'x'. Unmasked elements are left
   untouched.
*/
template <class T>
inline masked_matrix<T>& masked_matrix<T>::divideby(const matrix<T>& x)
   {
   // for A./B:
   // The size of A must be the same as the size of B.
   assert(x.m_xsize == m_data->m_xsize);
   assert(x.m_ysize == m_data->m_ysize);
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] /= x.m_data[i][j];
   return *this;
   }

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator+=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] += x;
   return *this;
   }

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator-=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] -= x;
   return *this;
   }

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator*=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] *= x;
   return *this;
   }

template <class T>
inline masked_matrix<T>& masked_matrix<T>::operator/=(const T x)
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] /= x;
   return *this;
   }

// user-defined operations

template <class T>
inline masked_matrix<T>& masked_matrix<T>::apply(T f(T))
   {
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            m_data->m_data[i][j] = f(m_data->m_data[i][j]);
   return *this;
   }

// information services

template <class T>
inline int masked_matrix<T>::size() const
   {
   int result = 0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            result++;
   return result;
   }

// statistical operations

template <class T>
inline T masked_matrix<T>::min() const
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

template <class T>
inline T masked_matrix<T>::max() const
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

template <class T>
inline T masked_matrix<T>::sum() const
   {
   assert(m_data->m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            result += m_data->m_data[i][j];
   return result;
   }

template <class T>
inline T masked_matrix<T>::sumsq() const
   {
   assert(m_data->m_xsize > 0);
   T result = 0;
   for(int i=0; i<m_data->m_xsize; i++)
      for(int j=0; j<m_data->m_ysize; j++)
         if(m_mask(i,j))
            result += m_data->m_data[i][j] * m_data->m_data[i][j];
   return result;
   }

template <class T>
inline T masked_matrix<T>::var() const
   {
   const T _mean = mean();
   const T _var = sumsq()/T(size()) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

}; // end namespace

#endif
