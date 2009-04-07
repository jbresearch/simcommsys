#ifndef __size_h
#define __size_h

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
   \brief   Size of templated parameter object.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

template <template<class> class T>
class size {
};


/*!
   \brief   Size specialization for vector.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \todo Consider hiding fields
*/

template <>
class size<vector> {
public:
   int  x;
public:
   explicit size(int x=0) { this->x = x; };
   operator int() const { return x; };
};


/*!
   \brief   Size specialization for matrix.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \todo Consider hiding fields
*/

template <>
class size<matrix> {
public:
   int  x;
   int  y;
public:
   explicit size(int x=0, int y=0) { this->x = x; this->y = y; };
   operator int() const { return x*y; };
};

}; // end namespace

#endif
