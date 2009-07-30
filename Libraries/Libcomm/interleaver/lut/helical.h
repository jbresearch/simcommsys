#ifndef __helical_h
#define __helical_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Helical Interleaver.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 */

template <class real>
class helical : public lut_interleaver<real> {
   int rows, cols;
protected:
   void init(const int tau, const int rows, const int cols);
   helical()
      {
      }
public:
   helical(const int tau, const int rows, const int cols)
      {
      init(tau, rows, cols);
      }
   ~helical()
      {
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(helical);
};

} // end namespace

#endif

