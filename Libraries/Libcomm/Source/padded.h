#ifndef __padded_h
#define __padded_h

#include "config.h"
#include "interleaver.h"
#include "onetimepad.h"
#include "serializer.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   Padded Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (27 Feb 1999)
  allowed client control of OTP (should it be terminated and renewable)

   \version 1.11 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance
  with interleaver 1.10)

   \version 1.20 (27 Feb 2002)
  moved contruction and transform functions into the implementation file rather than inline.
  added serialization facility (in accordance with interleaver 1.20).
  changed the definition of output() to be virtual.

   \version 1.21 (28 Feb 2002)
  changed output and serialize to be non-virtual functions - it doesn't quite
  make sense to keep these virtual while the transform functions are not.

   \version 1.22 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

   \version 1.30 (7 Mar - 8 Mar 2002)
  updated the system to conform with the completed serialization protocol for interleaver
  (in conformance with interleaver 1.32), by adding the necessary name() function, and
  also by removing the class name reading/writing in serialize(); this is now done only
  in the stream << and >> functions. serialize() assumes that the correct class is
  being read/written. We also add a static serializer member and initialize it with this
  class's name and the static constructor/loader (adding that too, together with the
  necessary private empty constructor). Also made the interleaver object a public base
  class, rather than a virtual public one, since this was affecting the transfer of
  virtual functions within the class (causing access violations).

  Also changed the member onetimepad object to a pointer to such an object. This allows
  us to create an empty "padded" class without access to onetimepad's default constructor
  (which is private for that class).

   \version 1.31 (8 Mar 2002)
  changed the name() function to use the serializer's name(), introduced in serializer
  1.03, in order to improve consistency between what's written on saving the class
  and what the class is classified as with serializer.

   \version 1.32 (10 Mar 2002)
  changed the otp member to be a pointer to interleaver, not a pointer to onetimepad.
  This was done to avoid converting from type interleaver* to onetimepad* when assigning
  the newly created object from the stream.

   \version 1.33 (11 Mar 2002)
  changed access level of default constructor to protected - this should facilitate
  the creation of derived classes. Also, changed createandload to an inline function
  create() which simply allocates a new object and returns its pointer. This makes the
  system compatible with the new serializer protocol, as defined in serializer 1.10.
  Added public cloning function, and made all parameters to the constructor const; the
  constructor clones those elements that it needs.

   \version 1.34 (14 Mar 2002)
  added the necessary copy constructor (this needs to clone the heap members).

   \version 1.40 (27 Mar 2002)
  changed descriptive output function to conform with interleaver 1.40.

   \version 1.50 (19 Apr 2005)
  added 'transform' and 'inverse' for matrices of type 'logreal', in accordance with
  interleaver 1.50.

   \version 1.51 (3 Aug 2006)
  modified functions 'transform' & 'inverse' to indicate within the prototype which
  parameters are input (by making them const), in accordance with interleaver 1.51.

   \version 1.60 (6 Nov 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.61 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class padded : public interleaver {
   static const libbase::serializer shelper;
   static void* create() { return new padded; };
   interleaver *otp;
   interleaver *inter;
protected:
   padded();
public:
   padded(const interleaver& inter, const fsm& encoder, const int tau, const bool terminated, const bool renewable);
   padded(const padded& x);
   ~padded();

   padded* clone() const { return new padded(*this); };
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
