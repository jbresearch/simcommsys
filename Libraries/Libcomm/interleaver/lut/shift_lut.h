#ifndef __shift_lut_h
#define __shift_lut_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"
#include <stdio.h>
#include <iostream>

namespace libcomm {

/*!
 * \brief   Barrel-Shifting LUT Interleaver.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 */

template <class real>
class shift_lut : public lut_interleaver<real> {
   int amount;
protected:
   void init(const int amount, const int tau);
   shift_lut()
      {
      }
public:
   shift_lut(const int amount, const int tau)
      {
      init(amount, tau);
      }
   ~shift_lut()
      {
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(shift_lut)
};

} // end namespace

#endif
