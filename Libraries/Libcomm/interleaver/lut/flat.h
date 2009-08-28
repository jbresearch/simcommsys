#ifndef __flat_h
#define __flat_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Flat Interleaver.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 */

template <class real>
class flat : public lut_interleaver<real> {
protected:
   void init(const int tau);
   flat()
      {
      }
public:
   flat(const int tau)
      {
      init(tau);
      }
   ~flat()
      {
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(flat);
};

} // end namespace

#endif

