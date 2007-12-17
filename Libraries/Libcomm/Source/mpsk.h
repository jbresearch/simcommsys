#ifndef __mpsk_h
#define __mpsk_h
      
#include "config.h"
#include "vcs.h"
#include "lut_modulator.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   M-PSK Modulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

  Version 1.10 (13 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with modulator 1.10), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too).

  Version 1.20 (27 Mar 2002)
  changed descriptive output function to conform with modulator 1.30.

  Version 2.00 (27 Mar 2002)
  renamed the class to mpsk, and modified it to create any M-PSK scheme. Also removed
  the qpsk class, since this is now redundant. Modifications requried include changing
  the creator to require a parameter (the number of modulation symbols), addition of
  a protected default constructor, inclusion of the m-parameter in serialization
  functions and in the description function, and addition of an initialization
  function (for use in loading & construction).

  Version 2.10 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
  
  Version 2.11 (24 Apr 2007)
  * fixed a bug in serialization code; final newline was not appended

  Version 2.12 (25 Oct 2007)
  * modified to comply with modulator 1.50 & lut_modulator 1.00
  * this class now inherits the functionality of lut_modulator
  * the LUT is now held in 'lut' rather than 'map'

  Version 2.13 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

class mpsk : public lut_modulator {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new mpsk; };
protected:
   mpsk() {};
   void init(const int m);
public:
   mpsk(const int m) { init(m); };
   ~mpsk() {};

   mpsk *clone() const { return new mpsk(*this); };             // cloning operation
   const char* name() const { return shelper.name(); };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
