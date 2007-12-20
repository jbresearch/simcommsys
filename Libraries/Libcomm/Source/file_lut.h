#ifndef __file_lut_h
#define __file_lut_h

#include "config.h"
#include "named_lut.h"

namespace libcomm {

/*!
   \brief   File-loaded LUT Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (31 Aug 1999)
  modified the file interleaver to allow JPL termination.

   \version 1.11 (4 Nov 2001)
  added a function which outputs details on the interleaving scheme (in accordance
  with interleaver 1.10)

   \version 1.12 (15 Nov 2001)
  moved creator/destroyer to .cpp instead of header file.

   \version 1.20 (28 Feb 2002)
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
  changed access level of default constructor to protected - this should facilitate
  the creation of derived classes. Also, changed createandload to an inline function
  create() which simply allocates a new object and returns its pointer. This makes the
  system compatible with the new serializer protocol, as defined in serializer 1.10.
  Also, added public cloning function.

   \version 1.33 (13 Mar 2002)
  changed base class to named_lut, which provides the necessary functions except
  creation.

   \version 1.40 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
*/

class file_lut : public named_lut {
public:
   file_lut(const char *filename, const int tau, const int m);
   ~file_lut() {};
};

}; // end namespace

#endif

