#ifndef __mapcc_h
#define __mapcc_h

#include "config.h"
#include "vcs.h"

#include "codec.h"
#include "fsm.h"
#include "bcjr.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

namespace libcomm {

/*!
   \brief   Maximum A-Posteriori Decoder.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.01 (5 Mar 1999)
   renamed the mem_order() function to tail_length(), which is more indicative of the true
   use of this function (complies with codec 1.01).

   \version 1.10 (7 Jun 1999)
   modified the system to comply with codec 1.10.

   \version 1.11 (24 Oct 2001)
   moved most functions to the cpp file rather than the header, in order to be able to
   compile the header with Microsoft Extensions. Naturally compilation is faster, but this
   also requires realizations of the class within the cpp file. This was done for mpreal,
   mpgnu and logreal.

   \version 1.12 (4 Nov 2001)
   added a function which outputs details on the codec (in accordance with codec 1.20)

   \version 1.13 (23 Feb 2002)
   added flushes to all end-of-line clog outputs, to clean up text user interface.

   \version 1.13 (6 Mar 2002)
   also changed use of iostream from global to std namespace.

   \version 1.14 (7 Mar 2002)
   renamed class from "map" to "mapcc" to avoid conflict with STL map. The new name
   was chosen to reflect that this is a codec for maximum-a-posteriori decoding of
   convolutional codes.

   \version 1.20 (13 Mar 2002)
   updated the system to conform with the completed serialization protocol (in conformance
   with codec 1.30), by adding the necessary name() function, and also by adding a static
   serializer member and initialize it with this class's name and the static constructor
   (adding that too, together with the necessary protected default constructor). Also made
   the codec object a public base class, rather than a virtual public one, since this
   was affecting the transfer of virtual functions within the class (causing access
   violations). Also made the version info a const member variable, and made all parameters
   of the constructor const; the constructor is then expected to clone all objects and
   the destructor is expected to delete them.  Also added a function which deallocates
   all heap variables.

   \version 1.30 (18 Mar 2002)
   modified to conform with codec 1.41.
   Also, removed the clog & cerr information output during initialization.

   \version 1.31 (23 Mar 2002)
   changed the definition of shelper - instead of a template definition, which uses the
   typeid.name() function to determine the name of the arithmetic class, we now manually
   define shelper for each instantiation. This avoids the ambiguity (due to implementation
   dependence) in name().

   \version 1.40 (27 Mar 2002)
   changed descriptive output function to conform with codec 1.50.

   \version 1.41 (17 Jul 2006)
   in translate, made an explicit conversion of the output of round to int, to conform
   with itfunc 1.07.

   \version 1.42 (6 Oct 2006)
   modified for compatibility with VS .NET 2005:
   - in translate, modified use of pow to avoid ambiguity

   \version 1.50 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.51 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]

   \version 1.52 (14 Dec 2007)
   - fixed minor error in description output
   - added flags for terminated and circular trellises
   - NOTE: there is no check for both being set!
*/

template <class real> class mapcc : public codec, private bcjr<real> {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new mapcc<real>; };
private:
   fsm      *encoder;
   double   rate;
   int      tau;           //!< Block length (including tail, if any)
   bool     endatzero;     //!< True for terminated trellis
   bool     circular;      //!< True for circular trellis
   int      m;             //!< encoder memory order
   int      M;             //!< Number of states
   int      K;             //!< Number of input combinations
   int      N;             //!< Number of output combinations
   libbase::matrix<double> R, ri, ro;   // BCJR statistics
protected:
   void init();
   void free();
   mapcc();
public:
   mapcc(const fsm& encoder, const int tau, const bool endatzero, const bool circular=false);
   ~mapcc() { free(); };

   mapcc *clone() const { return new mapcc(*this); };           // cloning operation
   const char* name() const { return shelper.name(); };

   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void translate(const libbase::matrix<double>& ptable);
   void decode(libbase::vector<int>& decoded);

   int block_size() const { return tau; };
   int num_inputs() const { return K; };
   int num_outputs() const { return N; };
   int tail_length() const { return m; };
   int num_iter() const { return 1; };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

