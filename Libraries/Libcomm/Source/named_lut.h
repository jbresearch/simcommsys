#ifndef __named_lut_h
#define __named_lut_h

#include "config.h"
#include "vcs.h"
#include "lut_interleaver.h"
#include "serializer.h"
#include <string>
#include <iostream>

/*
  Version 1.00 (13 Mar 2002)
  original version. Intended as a base class to implement any interleaver which is
  specified directly by its LUT, which is externally generated (say by Simulated Annealing
  or another such method), and has a name associated with it. This version was adapted
  from file_lut 1.32, and supports forced tails. Derivative classes only need to provide
  their own constructors and destructors, as necessary.

  Version 1.10 (27 Mar 2002)
  changed descriptive output function to conform with interleaver 1.40.

  Version 1.20 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.21 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

class named_lut : public lut_interleaver {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new named_lut; };
protected:
   std::string lutname;
   int m;
   named_lut() {};
public:
   named_lut* clone() const { return new named_lut(*this); };
   const char* name() const { return shelper.name(); };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

