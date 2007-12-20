#ifndef __lut_interleaver_h
#define __lut_interleaver_h

#include "config.h"
#include "interleaver.h"
#include "serializer.h"
#include "fsm.h"

namespace libcomm {

/*!
   \brief   Lookup Table Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (29 Aug 1999)
  introduced concept of forced tail interleavers (as in divs95)

   \version 1.11 (26 Oct 2001)
  added a virtual destroy function (see interleaver.h)

   \version 1.12 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance
  with interleaver 1.10)

   \version 1.20 (27 Feb 2002)
  moved transform functions into the implementation file rather than inline.
  added serialization facility (in accordance with interleaver 1.20)

   \version 1.21 (6 Mar 2002)
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

   \version 1.31 (8 Mar 2002)
  changed the name() function to use the serializer's name(), introduced in serializer
  1.03, in order to improve consistency between what's written on saving the class
  and what the class is classified as with serializer.

   \version 1.32 (11 Mar 2002)
  changed createandload to an inline function create() which simply allocates a new
  object and returns its pointer. This makes the system compatible with the new
  serializer protocol, as defined in serializer 1.10.

   \version 1.40 (11 Mar 2002)
  removed serialization facility and serialize/output functions; this class is only
  intended as a base class for LUT-based interleavers, and should not be instantiated
  directly.

   \version 1.50 (19 Apr 2005)
  added 'transform' and 'inverse' for matrices of type 'logreal', in accordance with
  interleaver 1.50.

   \version 1.51 (3 Aug 2006)
  modified functions 'transform' & 'inverse' to indicate within the prototype which
  parameters are input (by making them const), in accordance with interleaver 1.51.

   \version 1.60 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class lut_interleaver : public interleaver {
protected:
   lut_interleaver() {};
   static const int tail; // a special LUT entry to signify a forced tail
   libbase::vector<int> lut;
public:
   virtual ~lut_interleaver() {};

   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;
   void inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;
};

}; // end namespace

#endif

