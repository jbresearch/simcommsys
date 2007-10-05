#ifndef __channel_h
#define __channel_h

#include "config.h"
#include "vcs.h"
#include "vector.h"

#include "sigspace.h"

#include <iostream>
#include <string>

/*
  Version 1.01 (26 Oct 2001)
  added a virtual destroy function (see interleaver.h)

  Version 1.02 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

  Version 1.10 (13 Mar 2002)
  added a virtual function which outputs details on the channel, together with a stream
  << operator too. Also added serialization facility. Created serialize and stream << and
  >> functions to conform with the new serializer protocol, as defined in serializer 1.10.
  The stream << output function first writes the name of the derived class, then calls its
  serialize() to output the data. The name is obtained from the virtual name() function.
  The stream >> input function first gets the name from the stream, then (via
  serialize::call) creates a new object of the appropriate type and calls its serialize()
  function to get the relevant data. Also added cloning function.

  Version 1.20 (17 Mar 2002)
  added a function which corrupts a vector of signals (called transmit). This implements
  the separation of functions from the codec block (as defined in codec 1.40), since
  transmission depends only on the channel, it should be implemented here.

  Version 1.30 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

  Version 1.31 (13 Apr 2002)
  changed vector resizing operation in transmit() to use the new format (vector 1.50).

  Version 1.40 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libcomm {

class channel {
   static const libbase::vcs version;
public:
   virtual ~channel() {};                 // virtual destructor
   virtual channel *clone() const = 0;		// cloning operation
   virtual const char* name() const = 0;  // derived object's name

   virtual void seed(const libbase::int32u s) = 0;
   virtual void set_eb(const double Eb) = 0;
   virtual void set_snr(const double snr_db) = 0;
   virtual double get_snr() const = 0;

   virtual sigspace corrupt(const sigspace& s) = 0;
   virtual double pdf(const sigspace& tx, const sigspace& rx) const = 0;

   void transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx);

   // description output
   virtual std::string description() const = 0;
   // object serialization - saving
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   friend std::ostream& operator<<(std::ostream& sout, const channel* x);
   // object serialization - loading
   virtual std::istream& serialize(std::istream& sin) = 0;
   friend std::istream& operator>>(std::istream& sin, channel*& x);
};

}; // end namespace

#endif

