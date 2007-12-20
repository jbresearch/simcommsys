#ifndef __puncture_file_h
#define __puncture_file_h

#include "config.h"
#include "puncture.h"
#include "serializer.h"
#include <string>

namespace libcomm {

/*!
   \brief   Filed Puncturing System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (7 Jun 1999)
  initial version, implementing a puncturing pattern read from file.

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
  updated to conform with puncture 2.20.

   \version 2.30 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 2.31 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class puncture_file : public puncture {
   static const libbase::serializer shelper;
   static void* create() { return new puncture_file; };
private:
   libbase::matrix<bool> pattern;
protected:
   std::string filename;
   puncture_file() {};
public:
   puncture_file(const char *fname, const int tau, const int sets);
   ~puncture_file() {};

   puncture_file *clone() const { return new puncture_file(*this); };           // cloning operation
   const char* name() const { return shelper.name(); };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
