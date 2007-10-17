#ifndef __channel_h
#define __channel_h

#include "config.h"
#include "vcs.h"
#include "vector.h"

#include "randgen.h"
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
  
  Version 1.50 (16 Oct 2007)
  * refactored the class to simplify inheritance:
    - set_eb() and set_snr() are now defined in this class, and call compute_parameters(), which
    is defined in derived classes.
    - consequently, various variables have now moved to this class, together with their getters
    - random generator has also moved into this class, together with its seeding functions
  * updated channel model to allow for insertions and deletions, as well as substitution errors.
  
  Version 1.51 (16 Oct 2007)
  * refactored further to simplify inheritance:
    - serialization functions are no longer pure virtual; this removes the need for derived classes
      to supply these, unless there is something specific to serialize.
  
  Version 1.52 (17 Oct 2007)
  * started direct work on implementing support for insertion/deletion:
    - observed that the channel base function corrupt() is only called from within this class (in the
      implementation of transmit(); similarly, pdf() is only called from within the modulator base
      class, in the implementation of demodulate().
    - corrupt() function has been moved into protected space; transmit() has been made virtual, and
      the default implementation still makes use of the corrupt() function from derived classes.
      What this means in practice is that derived classes implementing a DMC can simply implement
      corrupt() and rely on this class to make transmit() available to clients. This is exactly as they
      are doing so far.
*/

namespace libcomm {

class channel {
   static const libbase::vcs version;
private:
   // channel paremeters
   double   Eb, No, snr_db;
   // internal helper functions
   void compute_noise();
protected:
   // objects used by the derived channel
   libbase::randgen  r;
   // handle functions
   virtual void compute_parameters(const double Eb, const double No) {};
   // channel handle functions
   virtual sigspace corrupt(const sigspace& s) = 0;
public:
   // object handling
   channel();                             // constructor
   virtual ~channel() {};                 // virtual destructor
   virtual channel *clone() const = 0;		// cloning operation
   virtual const char* name() const = 0;  // derived object's name

   // reset function for random generator
   void seed(const libbase::int32u s);

   // setting and getting overall channel SNR
   void set_eb(const double Eb);
   void set_snr(const double snr_db);
   double get_snr() const { return snr_db; };

   // channel functions:
   // base functions implemented by derived classes
   virtual double pdf(const sigspace& tx, const sigspace& rx) const = 0;
   // functions implemented through handle functions, or alternatively implemented by derived classes
   virtual void transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx);

   // description output
   virtual std::string description() const = 0;
   // object serialization - saving
   virtual std::ostream& serialize(std::ostream& sout) const;
   friend std::ostream& operator<<(std::ostream& sout, const channel* x);
   // object serialization - loading
   virtual std::istream& serialize(std::istream& sin);
   friend std::istream& operator>>(std::istream& sin, channel*& x);
};

}; // end namespace

#endif

