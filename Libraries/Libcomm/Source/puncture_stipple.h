#ifndef __puncture_stipple_h
#define __puncture_stipple_h

#include "config.h"
#include "puncture.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Stippled Puncturing System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements an odd/even puncturing system for turbo codes.
*/

class puncture_stipple : public puncture {
private:
   int tau, sets;
protected:
   void init(const int tau, const int sets);
   puncture_stipple() {};
public:
   puncture_stipple(const int tau, const int sets) { init(tau, sets); };
   ~puncture_stipple() {};

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(puncture_stipple)
};

}; // end namespace

#endif
