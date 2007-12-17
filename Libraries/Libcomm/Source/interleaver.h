#ifndef __interleaver_h
#define __interleaver_h

#include "config.h"
#include "matrix.h"
#include "vector.h"
#include "logrealfast.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Interleaver Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.01 (26 Oct 2001)
  Added empty virtual destroy function - this is required so that any derived interleavers
  classes can have objects successfully deleted in the following situation:
      interleaver *object = new derived_interleaver(parameters);
      delete object;
  If the interleaver class didn't have a virtual destroy function, there would be no
  mechanism for the delete call to actually translate to the derived_interleaver's
  destroy function. I realized this last night when I couldn't sleep...

  Version 1.10 (4 Nov 2001)
  added a virtual function which outputs details on the interleaving scheme (this was 
  only done before in a non-standard print routine). Added a stream << operator too.

  Version 1.20 (27 Feb 2002)
  added serialization facility

  Version 1.30 (6 Mar 2002)
  updated serialization facility for loading objects, using the serializer 1.00
  protocol.

  Version 1.31 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.32 (7 Mar 2002)
  formalised the serialization protocol: the stream << output function first writes
  the name of the derived class, then calls its serialize() to output the data. The
  name is obtained from the virtual name() function. The stream >> input function
  first gets the name from the stream, then (via serialize::call) creates a new 
  object of the appropriate type and calls its serialize() function to get the 
  relevant data. Also, changed the definition of stream << output to take the 
  pointer to the interleaver class directly, not by reference.

  Version 1.33 (11 Mar 2002)
  changed the stream >> function to conform with the new serializer protocol, as
  defined in serializer 1.10. Also added the requirement for providing a public
  cloning function.

  Version 1.40 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

  Version 1.50 (18 Apr - 19 Apr 2005)
  * tried to change the matrix transform and inverse function definitions to make them
  operate on a matrix of any class 'T'. This change was required to allow the construction
  of turbo codecs where the inter-iteration statistics are not of type 'double'.
  * however, such template functions cannot be virtualized; this leaves two design options:
  1) make 'interleaver' a template class; this is elegant in the sense that it transfers
  the choice of matrix type support to the creation of the class. It also allows the
  transform and inverse functions to be easily virtualized. However, it means that all
  derived classes need to be similarly templated. This solution is non-ideal because the
  class should not need to be templated (there is no internal difference between the
  various versions).
  2) add 'transform' and 'inverse' functions for all the required matrix types. This
  simplifies the implmentation because there is no need to templatize the class.
  However, it also leads to needless code duplication.
  In an ideal world, there would be a way to define the need for 'transform' and 'inverse'
  (as pure virtual functions in this class) that operate on matrix parameters of _any_ type.
  * option (2) was chosen; derived classes were changed accordingly.

  Version 1.51 (3 Aug 2006)
  modified functions 'transform' & 'inverse' to indicate within the prototype which
  parameters are input (by making them const). While this should not change any results,
  it is a forward step to simplify debugging, This was necessitated by turbo 2.47; all
  derived classes were also changed accordingly.

  Version 1.60 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class interleaver {
public:
   virtual ~interleaver() {};
   virtual interleaver* clone() const = 0;
   // derived class/object  name
   virtual const char* name() const = 0;
   // intra-frame operations
   virtual void seed(const int s) {};
   virtual void advance() {};
   // transform functions - note that 'in' and 'out' should NOT be the same!
   virtual void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const = 0;
   virtual void transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const = 0;
   virtual void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const = 0;
   // alternative matrix types for transform and inverse
   virtual void transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const = 0;
   virtual void inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const = 0;

   // description output
   virtual std::string description() const = 0;
   // object serialization - saving
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   friend std::ostream& operator<<(std::ostream& sout, const interleaver* x);
   // object serialization - loading
   virtual std::istream& serialize(std::istream& sin) = 0;
   friend std::istream& operator>>(std::istream& sin, interleaver*& x);
};

}; // end namespace

#endif
