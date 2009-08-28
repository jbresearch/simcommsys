#ifndef __file_lut_h
#define __file_lut_h

#include "config.h"
#include "interleaver/lut/named_lut.h"

namespace libcomm {

/*!
 * \brief   File-loaded LUT Interleaver.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 */

template <class real>
class file_lut : public named_lut<real> {
public:
   file_lut(const char *filename, const int tau, const int m);
   ~file_lut()
      {
      }
};

} // end namespace

#endif

