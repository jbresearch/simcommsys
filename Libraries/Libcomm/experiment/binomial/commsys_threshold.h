#ifndef __commsys_threshold_h
#define __commsys_threshold_h

#include "config.h"
#include "commsys_simulator.h"

namespace libcomm {

/*!
 \brief   Communication System Simulator - Variation of modem threshold.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 A variation on the regular commsys_simulator object, taking a fixed channel
 parameter and varying modem threshold.

 \todo Remove assumption of a dminner-derived modem.
 */
template <class S, class R = commsys_errorrates>
class commsys_threshold : public commsys_simulator<S, R> {
public:
   // Experiment parameter handling
   void set_parameter(const double x);
   double get_parameter() const;

   // Serialization Support
DECLARE_SERIALIZER(commsys_threshold);
};

} // end namespace

#endif
