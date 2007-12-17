#ifndef __puncture_stipple_h
#define __puncture_stipple_h

#include "config.h"
#include "vcs.h"
#include "puncture.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   .
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00 (7 Jun 1999)
  initial version, implementing an odd/even punctured system.

  Version 1.01 (4 Nov 2001)
  added a function which outputs details on the puncturing scheme (in accordance 
  with puncture 1.10)

  Version 1.02 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.10 (13 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with puncture 1.20), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too, together with the necessary protected default constructor). Also made
  the puncture object a public base class, rather than a virtual public one, since this
  was affecting the transfer of virtual functions within the class (causing access
  violations).

  Version 2.00 (18 Mar 2002)
  updated to conform with puncture 2.00.

  Version 2.10 (27 Mar 2002)
  changed descriptive output function to conform with puncture 2.10.

  Version 2.20 (27 Mar 2002)
  updated to conform with puncture 2.20.

  Version 2.30 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 2.31 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class puncture_stipple : public puncture {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new puncture_stipple; };
private:
   int tau, sets;
protected:
   void init(const int tau, const int sets);
   puncture_stipple() {};
public:
   puncture_stipple(const int tau, const int sets) { init(tau, sets); };
   ~puncture_stipple() {};

   puncture_stipple *clone() const { return new puncture_stipple(*this); };             // cloning operation
   const char* name() const { return shelper.name(); };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
