#ifndef __vale96int_h
#define __vale96int_h

#include "config.h"
#include "named_lut.h"

namespace libcomm {

/*!
 \brief   Matt Valenti's Interleaver.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 */

template <class real>
class vale96int : public named_lut<real> {
public:
   vale96int();
   ~vale96int()
      {
      }
};

} // end namespace

#endif
