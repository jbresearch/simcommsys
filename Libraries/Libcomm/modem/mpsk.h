#ifndef __mpsk_h
#define __mpsk_h

#include "config.h"
#include "lut_modulator.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   M-PSK Modulator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \note Gray code mapping is used for binary representation of
 * adjacent points on the constellation.
 */

class mpsk : public lut_modulator {
protected:
   mpsk()
      {
      }
   void init(const int m);
public:
   mpsk(const int m)
      {
      init(m);
      }
   ~mpsk()
      {
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(mpsk);
};

} // end namespace

#endif
