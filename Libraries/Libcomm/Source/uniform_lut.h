#ifndef __uniform_lut_h
#define __uniform_lut_h

#include "config.h"
#include "vcs.h"
#include "lut_interleaver.h"
#include "serializer.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   Uniform Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.10 (31 Aug 1999)
  modified the uniform interleaver to allow JPL termination.

  Version 1.11 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance 
  with interleaver 1.10)

  Version 1.20 (28 Feb 2002)
  added serialization facility (in accordance with interleaver 1.20)

  Version 1.21 (1 Mar 2002)   
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Here we chose to take the loop variables into function scope & also to use vectors.

  Version 1.22 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.30 (7 Mar - 8 Mar 2002)
  updated the system to conform with the completed serialization protocol for interleaver
  (in conformance with interleaver 1.32), by adding the necessary name() function, and
  also by removing the class name reading/writing in serialize(); this is now done only
  in the stream << and >> functions. serialize() assumes that the correct class is
  being read/written. We also add a static serializer member and initialize it with this
  class's name and the static constructor/loader (adding that too, together with the
  necessary private empty constructor). Also made the interleaver object a public base
  class, rather than a virtual public one, since this was affecting the transfer of
  virtual functions within the class (causing access violations).

  Version 1.31 (8 Mar 2002)
  changed the name() function to use the serializer's name(), introduced in serializer
  1.03, in order to improve consistency between what's written on saving the class
  and what the class is classified as with serializer. Also, added public cloning function.

  Version 1.40 (27 Mar 2002)
  changed descriptive output function to conform with interleaver 1.40.

  Version 1.50 (6 Nov 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.51 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class uniform_lut : public lut_interleaver {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new uniform_lut; };
   libbase::randgen  r;
   int tau, m;
protected:
   void init(const int tau, const int m);
   uniform_lut() {};
public:
   uniform_lut(const int tau, const int m) { init(tau, m); };
   ~uniform_lut() {};
   uniform_lut* clone() const { return new uniform_lut(*this); };
   const char* name() const { return shelper.name(); };

   void seed(const int s);
   void advance();

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace
   
#endif
