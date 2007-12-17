#ifndef __codec_h
#define __codec_h

#include "config.h"
#include "vcs.h"
#include "matrix.h"
#include "vector.h"
#include "itfunc.h"

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

  Version 1.01 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function.

  Version 1.10 (7 Jun 1999)
  modulated sequence changed from matrix to vector, in order to simplify the implementation
  of puncturing. Now, num_symbols returns the length of the signal space vector.

  Version 1.11 (26 Oct 2001)
  added a virtual destroy function (see interleaver.h)

  Version 1.20 (4 Nov 2001)
  added a virtual function which outputs details on the codec (this was only done before
  in the construction mechanism). Added a stream << operator too.

  Version 1.21 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

  Version 1.30 (13 Mar 2002)
  added serialization facility. Created serialize and stream << and >> functions to
  conform with the new serializer protocol, as defined in serializer 1.10. The stream
  << output function first writes the name of the derived class, then calls its
  serialize() to output the data. The name is obtained from the virtual name() function.
  The stream >> input function first gets the name from the stream, then (via
  serialize::call) creates a new object of the appropriate type and calls its serialize()
  function to get the relevant data. Also added cloning function.

  Version 1.40 (17 Mar 2002)
  added information function which returns the number of values each output symbol can
  take. Also removed the modulate/transmit/demodulate functions (which are now in the
  modulator and channel modules) and also num_symbols (since this is no longer necessary).
  Puncturing is now to be performed as a separate step, to avoid the overhead when there
  is no puncturing. Also, the num_iter function has no default value now (to make sure
  it's not overlooked). Finally, created a new function 'translate' which computes the
  necessary prabability tables for the particular codec from the probabilities of each
  modulation symbol as received from the channel. This function should be called before
  the first decode iteration for each block. Also, the input vector to encode has been
  made const.

  Version 1.41 (18 Mar 2002)
  added three informative functions, giving the number of input bits & output bits per
  frame, and the data rate for the code. These can be computed from the other information
  functions for all codes, so they have been implemented as part of the base class. Note
  that the number of bits need not be an integer (if the codes are not binary).
  Also made all information functions const. Finally, changed the input vector for
  encode() back to non-const (tail entries need to be filled in by the encoder with the
  actual values used).

  Version 1.50 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

  Version 1.60 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class codec {
   static const libbase::vcs version;
public:
   virtual ~codec() {};                   // virtual destructor
   virtual codec *clone() const = 0;       // cloning operation
   virtual const char* name() const = 0;  // derived object's name

   virtual void seed(const int s) {};

   virtual void encode(libbase::vector<int>& source, libbase::vector<int>& encoded) = 0;
   virtual void translate(const libbase::matrix<double>& ptable) = 0;
   virtual void decode(libbase::vector<int>& decoded) = 0;

   virtual int block_size() const = 0;
   virtual int num_inputs() const = 0;
   virtual int num_outputs() const = 0;
   virtual int tail_length() const = 0;
   virtual int num_iter() const = 0;

   double input_bits() const { return libbase::log2(num_inputs())*(block_size() - tail_length()); };
   double output_bits() const { return libbase::log2(num_outputs())*block_size(); };
   double rate() const { return input_bits()/output_bits(); };

   // description output
   virtual std::string description() const = 0;
   // object serialization - saving
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   friend std::ostream& operator<<(std::ostream& sout, const codec* x);
   // object serialization - loading
   virtual std::istream& serialize(std::istream& sin) = 0;
   friend std::istream& operator>>(std::istream& sin, codec*& x);
};

}; // end namespace

#endif

