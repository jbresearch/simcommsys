#ifndef __commsys_threshold_h
#define __commsys_threshold_h

#include "config.h"
#include "commsys.h"

namespace libcomm {

/*!
   \brief   Communication System - Variation of modem threshold.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   A variation on the regular commsys_simulator object, taking a fixed channel
   parameter and varying modem threshold (currently assumes a dminner-derived
   modem).
*/
template <class S, class R=commsys_errorrates>
class commsys_threshold : public commsys_simulator<S,R> {
public:
   // Experiment parameter handling
   void set_parameter(const double x);
   double get_parameter() const;

   // Serialization Support
   DECLARE_SERIALIZER(commsys_threshold)
};

}; // end namespace

#endif
