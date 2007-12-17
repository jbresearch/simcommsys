#ifndef __vale96int_h
#define __vale96int_h

#include "config.h"
#include "vcs.h"
#include "named_lut.h"

namespace libcomm {

/*!
   \brief   Matt Valenti's Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.01 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance 
  with interleaver 1.10)

  Version 1.10 (28 Feb 2002)
  moved creation functions into the implementation file rather than inline.
  added serialization facility (in accordance with interleaver 1.20).
  The use of this class is essentially deprecated - when serializing, it is marked
  with the same marker as file_lut. Creation of similar classes is also discouraged
  (it's better to created interleavers directly from files).

  Version 1.11 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.20 (7 Mar - 8 Mar 2002)
  updated the system to conform with the completed serialization protocol for interleaver
  (in conformance with interleaver 1.32), by adding the necessary name() function, and
  also by removing the class name reading/writing in serialize(); this is now done only
  in the stream << and >> functions. serialize() assumes that the correct class is
  being read/written. We also add a static serializer member and initialize it with this
  class's name and the static constructor/loader (adding that too, together with the
  necessary private empty constructor). Also made the interleaver object a public base
  class, rather than a virtual public one, since this was affecting the transfer of
  virtual functions within the class (causing access violations).

  Version 1.21 (8 Mar 2002)
  changed the name() function to use the serializer's name(), introduced in serializer
  1.03, in order to improve consistency between what's written on saving the class
  and what the class is classified as with serializer.

  Version 1.22 (11 Mar 2002)
  changed access level of init() to protected - this should facilitate the creation of
  derived classes. Also, changed createandload to an inline function create() which
  simply allocates a new object and returns its pointer. This makes the system
  compatible with the new serializer protocol, as defined in serializer 1.10.
  Also, added public cloning function.

  Version 1.23 (13 Mar 2002)
  changed base class to named_lut, which provides the necessary functions except
  creation.

  Version 1.30 (6 Nov 2006)
  * defined class and associated data within "libcomm" namespace.
*/

class vale96int : public named_lut {
   static const libbase::vcs version;
public:
   vale96int();
   ~vale96int() {};
};

}; // end namespace

#endif
