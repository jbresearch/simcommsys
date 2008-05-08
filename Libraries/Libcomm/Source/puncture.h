#ifndef __puncture_h
#define __puncture_h

#include "config.h"
#include "serializer.h"
#include "matrix.h"
#include "vector.h"
#include "sigspace.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Base Puncturing System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (7 Jun 1999)
  initial version, base class with three implementations (unpunctured, odd/even, and from file).

   \version 1.10 (4 Nov 2001)
  added a virtual function which outputs details on the puncturing scheme (this was
  only done before in a non-standard print routine). Added a stream << operator too.

   \version 1.11 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

   \version 1.20 (13 Mar 2002)
  added serialization facility. Created serialize and stream << and >> functions to
  conform with the new serializer protocol, as defined in serializer 1.10. The stream
  << output function first writes the name of the derived class, then calls its
  serialize() to output the data. The name is obtained from the virtual name() function.
  The stream >> input function first gets the name from the stream, then (via
  serialize::call) creates a new object of the appropriate type and calls its serialize()
  function to get the relevant data. Also added cloning function.
  Added init() function, which fills in all member variables from the pattern matrix.

   \version 2.00 (18 Mar 2002)
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

   \version 2.10 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

   \version 2.20 (27 Mar 2002)
  removed the pattern matrix from a member - it is now passed (by reference) to the
  init() function, which is the only place where it is needed. Also removed the count,
  tau and s members, and replaced them by the number of inputs and the number of
  outputs; similarly, num_symbols, get_length and get_sets have been replaced by
  get_inputs and get_outputs.

   \version 2.30 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class puncture {
private:
   int inputs, outputs;
   libbase::vector<int> pos;
protected:
   void init(const libbase::matrix<bool>& pattern);  // fill-in all members from the pattern matrix
public:
   virtual ~puncture() {};                // virtual destructor

   void transform(const libbase::vector<sigspace>& in, libbase::vector<sigspace>& out) const;
   void inverse(const libbase::vector<sigspace>& in, libbase::vector<sigspace>& out) const;
   void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;

   double rate() const { return double(outputs)/double(inputs); };
   int get_inputs() const { return inputs; };
   int get_outputs() const { return outputs; };

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(puncture)
};

}; // end namespace

#endif
