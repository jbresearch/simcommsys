#ifndef __stream_lut_h
#define __stream_lut_h

#include "config.h"
#include "named_lut.h"
#include <stdio.h>

namespace libcomm {

/*!
 \brief   Stream-loaded LUT Interleaver.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 */

template <class real>
class stream_lut : public named_lut<real> {
public:
   stream_lut(const char *filename, FILE *file, const int tau, const int m);
   ~stream_lut()
      {
      }
};

} // end namespace

#endif
