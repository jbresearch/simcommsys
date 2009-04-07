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
*/

template <>
class size<vector> {
public:
   int  x;
};


/*!
   \brief   Size specialization for matrix.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

template <>
class size<matrix> {
public:
   int  x;
   int  y;
};

}; // end namespace

#endif
