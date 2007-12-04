#ifndef __uncoded_h
#define __uncoded_h

#include "config.h"
#include "vcs.h"

#include "codec.h"
#include "fsm.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

/*
  Version 1.01 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function (complies with codec 1.01).

  Version 1.10 (7 Jun 1999)
  modified the system to comply with codec 1.10.

  Version 1.02 (4 Nov 2001)
  added a function which outputs details on the codec (in accordance with codec 1.20)

  Version 1.03 (23 Feb 2002)
  added flushes to all end-of-line clog outputs, to clean up text user interface.

  Version 1.04 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.10 (13 Mar 2002)
  updated the system to conform with the completed serialization protocol (in conformance
  with codec 1.30), by adding the necessary name() function, and also by adding a static
  serializer member and initialize it with this class's name and the static constructor
  (adding that too, together with the necessary protected default constructor). Also made
  the codec object a public base class, rather than a virtual public one, since this
  was affecting the transfer of virtual functions within the class (causing access
  violations). Also added a function which deallocates all heap variables.

  Version 1.20 (17 Mar 2002)
  modified to conform with codec 1.40.
  Also, removed the clog & cerr information output during initialization.

  Version 1.21 (18 Mar 2002)
  modified to conform with codec 1.41.

  Version 1.30 (27 Mar 2002)
  changed descriptive output function to conform with codec 1.50.

  Version 1.31 (17 Jul 2006)
  in translate, made an explicit conversion of round's output to int, to conform with the
  changes in itfunc 1.07.

  Version 1.32 (6 Oct 2006)
  modified for compatibility with VS .NET 2005:
  * in modulate, modified use of pow to avoid ambiguity

  Version 1.40 (6 Nov 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.41 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

class uncoded : public codec {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new uncoded; };
   fsm             *encoder;
   int                  tau;            // block length
   int                  K, N;              // # of inputs and outputs (respectively)
   libbase::vector<int> lut;
   libbase::matrix<double> R;
protected:
   void init();
   void free();
   uncoded();
public:
   uncoded(const fsm& encoder, const int tau);
   ~uncoded() { free(); };

   uncoded *clone() const { return new uncoded(*this); };               // cloning operation
   const char* name() const { return shelper.name(); };

   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void translate(const libbase::matrix<double>& ptable);
   void decode(libbase::vector<int>& decoded);

   int block_size() const { return tau; };
   int num_inputs() const { return K; };
   int num_outputs() const { return N; };
   int tail_length() const { return 0; };
   int num_iter() const { return 1; };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

