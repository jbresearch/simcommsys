#ifndef __puncture_null_h
#define __puncture_null_h

#include "config.h"
#include "puncture.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Null Puncturing System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (7 Jun 1999)
  initial version, implementing an unpunctured system.

   \version 1.01 (4 Nov 2001)
  added a function which outputs details on the puncturing scheme (in accordance
  with puncture 1.10)

   \version 1.02 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

   \version 1.10 (13 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with puncture 1.20), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too, together with the necessary protected default constructor). Also made
  the puncture object a public base class, rather than a virtual public one, since this
  was affecting the transfer of virtual functions within the class (causing access
  violations).

   \version 2.00 (18 Mar 2002)
  updated to conform with puncture 2.00.

   \version 2.10 (27 Mar 2002)
  changed descriptive output function to conform with puncture 2.10.

   \version 2.20 (27 Mar 2002)
  updated to conform with puncture 2.20; changed the init() function and constructor
  to only require one parameter.

   \version 2.30 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.

   \version 2.31 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class puncture_null : public puncture {
protected:
   void init(const int tau);
   puncture_null() {};
public:
   puncture_null(const int tau) { init(tau); };
   ~puncture_null() {};

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(puncture_null)
};

}; // end namespace

#endif
