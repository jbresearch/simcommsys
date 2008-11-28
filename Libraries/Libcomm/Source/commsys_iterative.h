#ifndef __commsys_iterative_h
#define __commsys_iterative_h

#include "commsys.h"

namespace libcomm {

/*!
   \brief   General Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   General templated commsys, directly derived from common base.
*/

template <class S, template<class> class C=libbase::vector>
class commsys_iterative : public commsys<S,C> {
public:
   // Serialization Support
   DECLARE_SERIALIZER(commsys_iterative);
};

}; // end namespace

#endif
