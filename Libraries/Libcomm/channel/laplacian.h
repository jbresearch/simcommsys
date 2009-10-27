#ifndef __laplacian_h
#define __laplacian_h

#include "config.h"
#include "channel.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

namespace libcomm {

/*!
 * \brief   Additive Laplacian Noise Channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \version 1.00 (15 Apr 2001)
 * First version - note that as with the Gaussian channel, the distribution
 * has zero mean even in this case.
 *
 * \version 1.01 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 *
 * \version 1.10 (13 Mar 2002)
 * updated the system to conform with the completed serialization protocol (in conformance
 * with channel 1.10), by adding the necessary name() function, and also by adding a static
 * serializer member and initialize it with this class's name and the static constructor
 * (adding that too). Also made the channel object a public base class, rather than a
 * virtual public one, since this was affecting the transfer of virtual functions within
 * the class (causing access violations). Also moved most functions into the implementation
 * file rather than here.
 *
 * \version 1.20 (27 Mar 2002)
 * changed descriptive output function to conform with channel 1.30.
 *
 * \version 1.30 (30 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 *
 * \version 1.40 (16 Oct 2007)
 * changed class to conform with channel 1.50.
 *
 * \version 1.41 (16 Oct 2007)
 * changed class to conform with channel 1.51.
 *
 * \version 1.42 (17 Oct 2007)
 * changed class to conform with channel 1.52.
 *
 * \version 1.43 (29 Oct 2007)
 * - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
 *
 * \version 1.44 (24 Jan 2008)
 * - Changed derivation from channel to channel<sigspace>
 */

class laplacian : public channel<sigspace> {
   // channel paremeters
   double lambda;
private:
   // internal helper functions
   double f(const double x) const
      {
      return 1 / (2 * lambda) * exp(-fabs(x) / lambda);
      }
   double Finv(const double y) const
      {
      return (y < 0.5) ? lambda * log(2 * y) : -lambda * log(2 * (1 - y));
      }
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(laplacian)
};

} // end namespace

#endif

