#ifndef __diffturbo_h
#define __diffturbo_h

#include "config.h"
#include "serializer.h"

#include "turbo.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Diffused-Input Turbo Decoder.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.10 (8 Jun 1999)
  modified to comply with turbo 1.60 (new handling of puncturing)

  Version 2.00 (1 Jul 1999)
  changed the way we do modulo-2 additions with probability values.

  Version 2.10 (2 Jul 1999)
  started normalising statistics before doing the mod-2 additions.

  Version 2.20 (6 Jun 2000)
  modified to comply with turbo 1.80

  Version 2.21 (24 Oct 2001)
  moved most functions to the cpp file rather than the header, in order to be able to
  compile the header with Microsoft Extensions. Naturally compilation is faster, but this
  also requires realizations of the class within the cpp file. This was done for mpreal,
  mpgnu and logreal.

  Version 2.22 (4 Nov 2001)
  added a function which outputs details on the codec (in accordance with codec 1.20)

  Version 2.23 (1 Mar 2002)   
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Here we chose to take the loop variables into function scope.

  Version 2.30 (19 Mar 2002)
  modified to conform with codec 1.41.
  In the process, removed the use of puncturing from within this class, and also
  made the turbo<real> base public, not public virtual.
  Also, removed the clog & cerr information output during initialization.
  updated the system to conform with the completed serialization protocol (in conformance
  with codec 1.30), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too, together with the necessary protected default constructor). Also made
  the codec object a public base class, rather than a virtual public one, since this
  was affecting the transfer of virtual functions within the class (causing access
  violations). Also made the version info a const member variable, and made all parameters
  of the constructor const; the constructor is then expected to clone all objects and
  the destructor is expected to delete them. The only exception here is that the set
  of interleavers is passed as a vector of pointers. These must be allocated on the heap
  by the user, and will then be deleted by this class. Also added a function which
  deallocates all heap variables.

  Version 2.31 (23 Mar 2002)
  changed the definition of shelper - instead of a template definition, which uses the
  typeid.name() function to determine the name of the arithmetic class, we now manually
  define shelper for each instantiation. This avoids the ambiguity (due to implementation
  dependence) in name().

  Version 2.40 (27 Mar 2002)
  changed descriptive output function to conform with codec 1.50.

  Version 2.41 (12 Jun 2002)
  included <stdio.h> in the implementation file, which is needed for C-style file access.

  Version 2.42 (17 Jul 2006)
  updated for compilation with gcc, which has tighter control of templates & scoping; 
  in particular, things like calls to tail_length() have now been specifically given as
  turbo<real>::tail_length(). Alternatively, this->tail_length() may also have been used.

  Version 2.50 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 2.51 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

template <class real> class diffturbo : public turbo<real> {
   static const libbase::serializer shelper;
   static void* create() { return new diffturbo<real>; };
private:
   std::string filename;
   libbase::vector<int> lut;
   libbase::vector<int> source2, source3;
   libbase::matrix<double> decoded2, decoded3;
   void load_lut(const char *filename, const int tau);
   void add(libbase::matrix<double>& z, libbase::matrix<double>& x, libbase::matrix<double>& y, int zp, int xp, int yp);
protected:
   void init();
   diffturbo() {};
public:
   diffturbo(const char *filename, fsm& encoder, const int tau, libbase::vector<interleaver *>& inter, \
      const int iter, const bool simile, const bool endatzero, const bool parallel=false);
   ~diffturbo() {};
  
   diffturbo *clone() const { return new diffturbo(*this); };           // cloning operation
   const char* name() const { return shelper.name(); };

   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void decode(libbase::vector<int>& decoded);

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
