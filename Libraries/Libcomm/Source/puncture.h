#ifndef __puncture_h
#define __puncture_h

#include "config.h"
#include "vcs.h"
#include "matrix.h"
#include "vector.h"
#include "sigspace.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   .
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00 (7 Jun 1999)
  initial version, base class with three implementations (unpunctured, odd/even, and from file).

  Version 1.10 (4 Nov 2001)
  added a virtual function which outputs details on the puncturing scheme (this was 
  only done before in a non-standard print routine). Added a stream << operator too.

  Version 1.11 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.20 (13 Mar 2002)
  added serialization facility. Created serialize and stream << and >> functions to
  conform with the new serializer protocol, as defined in serializer 1.10. The stream
  << output function first writes the name of the derived class, then calls its
  serialize() to output the data. The name is obtained from the virtual name() function.
  The stream >> input function first gets the name from the stream, then (via
  serialize::call) creates a new object of the appropriate type and calls its serialize()
  function to get the relevant data. Also added cloning function.
  Added init() function, which fills in all member variables from the pattern matrix.

  Version 2.00 (18 Mar 2002)
  revamped puncturing class - the pattern matrix is now a tau by s matrix of bool rather
  than an s by tau matrix of int. Second, the pos matrix and the transmit() and position()
  functions have been removed (in favour of a better architecture for using puncture).
  More importantly, functions have been added that perform the puncturing and unpuncturing
  on a vector of sigspace elements or a matrix of probabilities (for unpuncturing only).
  This brings puncture in conformance with the new codec 1.41, and is achieved with the
  help of a vectors, pos(i), which will give the index in the unpunctured vector where
  element i from the punctured vector should be placed (it will always go somewhere). All
  remaining elements in the unpunctured vector should be set to 'equiprobable'. When
  performing the inverse puncturing on a matrix of probabilities, this means setting
  all probabilities for such entries to 1/M (where M is the number of modulation symbols,
  and hence the number of probability values in each entry). When working with a vector
  of sigspace, we set that signal to (0,0), assuming that the no-energy position is 
  equally distant from all modulation symbols (this is not true when the amplitude is
  also modulated, such as in QAM). Also made most data members private (except pattern,
  which needs to be set in the constructor of each implementation).

  Version 2.10 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

  Version 2.20 (27 Mar 2002)
  removed the pattern matrix from a member - it is now passed (by reference) to the
  init() function, which is the only place where it is needed. Also removed the count,
  tau and s members, and replaced them by the number of inputs and the number of
  outputs; similarly, num_symbols, get_length and get_sets have been replaced by
  get_inputs and get_outputs.

  Version 2.30 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class puncture {
   static const libbase::vcs version;
private:
   int inputs, outputs;
   libbase::vector<int> pos;
protected:
   void init(const libbase::matrix<bool>& pattern);  // fill-in all members from the pattern matrix
public:
   virtual ~puncture() {};                // virtual destructor
   virtual puncture *clone() const = 0; // cloning operation
   virtual const char* name() const = 0;  // derived object's name

   void transform(const libbase::vector<sigspace>& in, libbase::vector<sigspace>& out) const;
   void inverse(const libbase::vector<sigspace>& in, libbase::vector<sigspace>& out) const;
   void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;

   double rate() const { return double(outputs)/double(inputs); };
   int get_inputs() const { return inputs; };
   int get_outputs() const { return outputs; };

   // description output
   virtual std::string description() const = 0;
   // object serialization - saving
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   friend std::ostream& operator<<(std::ostream& sout, const puncture* x);
   // object serialization - loading
   virtual std::istream& serialize(std::istream& sin) = 0;
   friend std::istream& operator>>(std::istream& sin, puncture*& x);
};

}; // end namespace

#endif
