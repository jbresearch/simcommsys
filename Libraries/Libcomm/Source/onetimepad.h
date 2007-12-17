#ifndef __onetimepad_h
#define __onetimepad_h

#include "config.h"
#include "interleaver.h"
#include "serializer.h"
#include "fsm.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   One Time Pad Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.10 (27 Feb 1999)
  allowed client control of interleaver (should it be terminated and renewable)

  Version 1.11 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance 
  with interleaver 1.10)

  Version 1.12 (23 Feb 2002)
  added flushes to all end-of-line clog outputs, to clean up text user interface.

  Version 1.20 (27 Feb 2002)
  moved transform functions into the implementation file rather than inline.
  added serialization facility (in accordance with interleaver 1.20).
  changed the definition of output() to be virtual.

  Version 1.21 (28 Feb 2002)
  changed output and serialize to be non-virtual functions - it doesn't quite
  make sense to keep these virtual while the transform functions are not.

  Version 1.21 (1 Mar 2002)   
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Here we chose to take the loop variables into function scope.

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
  and what the class is classified as with serializer.

  Version 1.32 (11 Mar 2002)
  changed access level of default constructor to protected - this should facilitate
  the creation of derived classes. Also, changed createandload to an inline function
  create() which simply allocates a new object and returns its pointer. This makes the
  system compatible with the new serializer protocol, as defined in serializer 1.10.
  Added public cloning function, and made all parameters to the constructor const; the
  constructor clones those elements that it needs. Also made the default constructor
  initialize the member pointers to NULL.

  Version 1.33 (14 Mar 2002)
  added the necessary copy constructor (this needs to clone the heap members).

  Version 1.40 (27 Mar 2002)
  changed descriptive output function to conform with interleaver 1.40.

  Version 1.50 (19 Apr 2005)
  added 'transform' and 'inverse' for matrices of type 'logreal', in accordance with
  interleaver 1.50.

  Version 1.51 (3 Aug 2006)
  modified functions 'transform' & 'inverse' to indicate within the prototype which
  parameters are input (by making them const), in accordance with interleaver 1.51.

  Version 1.60 (6 Nov 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.61 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class onetimepad : public interleaver {
   static const libbase::serializer shelper;
   static void* create() { return new onetimepad; };
   bool terminated, renewable;
   fsm *encoder;
   int m, K;
   libbase::vector<int> pad;
   libbase::randgen r;
protected:
   onetimepad();
public:
   onetimepad(const fsm& encoder, const int tau, const bool terminated, const bool renewable);
   onetimepad(const onetimepad& x);
   ~onetimepad();
   onetimepad* clone() const { return new onetimepad(*this); };
   const char* name() const { return shelper.name(); };

   void seed(const int s);
   void advance();

   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;
   void inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
