#ifndef __rand_lut_h
#define __rand_lut_h

#include "config.h"
#include "lut_interleaver.h"
#include "serializer.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   Random LUT Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.01 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance
  with interleaver 1.10)

   \version 1.10 (28 Feb 2002)
  moved creation functions into the implementation file rather than inline.
  added serialization facility (in accordance with interleaver 1.20)

   \version 1.11 (1 Mar 2002)
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Rather than taking the loop variables into function scope, we chose to avoid having
  more than one loop per function, by using better methods.

   \version 1.12 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

   \version 1.20 (7 Mar - 8 Mar 2002)
  updated the system to conform with the completed serialization protocol for interleaver
  (in conformance with interleaver 1.32), by adding the necessary name() function, and
  also by removing the class name reading/writing in serialize(); this is now done only
  in the stream << and >> functions. serialize() assumes that the correct class is
  being read/written. We also add a static serializer member and initialize it with this
  class's name and the static constructor/loader (adding that too, together with the
  necessary private empty constructor). Also made the interleaver object a public base
  class, rather than a virtual public one, since this was affecting the transfer of
  virtual functions within the class (causing access violations).

   \version 1.21 (8 Mar 2002)
  changed the name() function to use the serializer's name(), introduced in serializer
  1.03, in order to improve consistency between what's written on saving the class
  and what the class is classified as with serializer.

   \version 1.22 (11 Mar 2002)
  changed access level of init() and default constructor to protected - this should
  facilitate the creation of derived classes. Also, changed createandload to an inline
  function create() which simply allocates a new object and returns its pointer. This
  makes the system compatible with the new serializer protocol, as defined in
  serializer 1.10. Also, added public cloning function.

   \version 1.23 (13 Mar 2002)
  fixed a bug in the creation function - since this is a random simile interleaver, there
  is a restriction that the interleaver size must be a multiple of p, where p is the
  length of the encoder impulse response (cf my MPhil p.40). The constructor now gives an
  error if this is not the case.

   \version 1.30 (27 Mar 2002)
  changed descriptive output function to conform with interleaver 1.40.

   \version 1.40 (6 Nov 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.41 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class rand_lut : public lut_interleaver {
   static const libbase::serializer shelper;
   static void* create() { return new rand_lut; };
   int      p;
   libbase::randgen  r;
protected:
   void init(const int tau, const int m);
   rand_lut() {};
public:
   rand_lut(const int tau, const int m) { init(tau, m); };
   ~rand_lut() {};
   rand_lut* clone() const { return new rand_lut(*this); };
   const char* name() const { return shelper.name(); };

   void seed(const int s);
   void advance();

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
