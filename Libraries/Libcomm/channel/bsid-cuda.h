#ifndef __bsid_cuda_h
#define __bsid_cuda_h

#include "vector.h"
#include "matrix.h"
#include <iostream>

namespace cuda {

void query_devices(std::ostream& sout);

/*!
 * \brief   A single value in device memory - interface
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class value_interface {
protected:
   T* data;
protected:
   /*! \brief copy constructor
    * \note protect to disallow copying except through derived objects
    * \note makes a shallow copy
    */
   value_interface(const value_interface<T>& x)
      {
      data = x.data;
      }
   /*! \brief copy assignment
    * \note protect to disallow copying except through derived objects
    * \note makes a shallow copy
    */
   value_interface<T>& operator=(const value_interface<T>& x)
      {
      data = x.data;
      return *this;
      }
   // protect default constructor to only allow instantiation of derived objects
   //! default constructor
   value_interface() :
      data(NULL)
      {
      }
public:
};

/*!
 * \brief   A single value in device memory
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class value : public value_interface<T> {
private:
   typedef value_interface<T> base;
public:
   //! default constructor
   value()
      {
      }
   //! destructor
   ~value()
      {
      if (base::data)
         free();
      }
   //! copy constructor
   value(const value<T>& x);
   //! copy assignment
   value<T>& operator=(const value<T>& x);
   //! allocate element
   void allocate();
   //! free memory
   void free();
   //! copy from standard value
   value<T>& operator=(const T& x);
   //! copy to standard value
   operator T();
};

/*!
 * \brief   A one-dimensional array in device memory - interface
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class vector_interface {
protected:
   T* data;
   int length;
protected:
   /*! \brief copy constructor
    * \note protect to disallow copying except through derived objects
    * \note makes a shallow copy
    */
   vector_interface(const vector_interface<T>& x)
      {
      data = x.data;
      length = x.length;
      }
   /*! \brief copy assignment
    * \note protect to disallow copying except through derived objects
    * \note makes a shallow copy
    */
   vector_interface<T>& operator=(const vector_interface<T>& x)
      {
      data = x.data;
      length = x.length;
      return *this;
      }
   // protect default constructor to only allow instantiation of derived objects
   //! default constructor
   vector_interface() :
      data(NULL), length(0)
      {
      }
public:
   //! Total number of elements
   int size() const
      {
      return length;
      }
};

/*!
 * \brief   A one-dimensional array in device memory
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class vector : public vector_interface<T> {
private:
   typedef vector_interface<T> base;
public:
   //! default constructor
   vector()
      {
      }
   //! destructor
   ~vector()
      {
      if (base::data)
         free();
      }
   //! copy constructor
   vector(const vector<T>& x);
   //! copy assignment
   vector<T>& operator=(const vector<T>& x);
   //! allocate requested number of elements
   void allocate(int n);
   //! free memory
   void free();
   //! copy from standard vector
   vector<T>& operator=(const libbase::vector<T>& x);
   //! copy to standard vector
   operator libbase::vector<T>();
};

/*!
 * \brief   A two-dimensional array in device memory - interface
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class matrix_interface {
protected:
   T* data;
   int rows;
   int cols;
   size_t pitch; //!< in elements
protected:
   /*! \brief copy constructor
    * \note protect to disallow copying except through derived objects
    * \note makes a shallow copy
    */
   matrix_interface(const matrix_interface<T>& x)
      {
      data = x.data;
      rows = x.rows;
      cols = x.cols;
      pitch = x.pitch;
      }
   /*! \brief copy assignment
    * \note protect to disallow copying except through derived objects
    * \note makes a shallow copy
    */
   matrix_interface<T>& operator=(const matrix_interface<T>& x)
      {
      data = x.data;
      rows = x.rows;
      cols = x.cols;
      pitch = x.pitch;
      return *this;
      }
   // protect default constructor to only allow instantiation of derived objects
   //! default constructor
   matrix_interface() :
      data(NULL), rows(0), cols(0), pitch(0)
      {
      }
public:
};

/*!
 * \brief   A two-dimensional array in device memory
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class T>
class matrix : public matrix_interface<T> {
private:
   typedef matrix_interface<T> base;
public:
   //! default constructor
   matrix()
      {
      }
   //! destructor
   ~matrix()
      {
      if (base::data)
         free();
      }
   //! copy constructor
   matrix(const matrix<T>& x);
   //! copy assignment
   matrix<T>& operator=(const matrix<T>& x);
   //! allocate requested number of elements
   void allocate(int m, int n);
   //! free memory
   void free();
   //! copy from standard matrix
   matrix<T>& operator=(const libbase::matrix<T>& x);
   //! copy to standard matrix
   operator libbase::matrix<T>();
};

float bsid_receive(const vector<bool>& tx, const vector<bool>& rx,
      const matrix<float>& Rtable, const value<float>& Rval, const int I,
      const int xmax, const int N);

} // end namespace

#endif
