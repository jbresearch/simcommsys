#ifndef __rectangular_h
#define __rectangular_h

#include "config.h"
#include "lut_interleaver.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Rectangular Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

*/

template <class real>
class rectangular : public lut_interleaver<real> {
   int rows, cols;
protected:
   void init(const int tau, const int rows, const int cols);
   rectangular() {};
public:
   rectangular(const int tau, const int rows, const int cols) { init(tau, rows, cols); };
   ~rectangular() {};

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(rectangular);
};

}; // end namespace

#endif
