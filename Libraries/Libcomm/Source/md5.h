#ifndef __md5_h
#define __md5_h

#include "config.h"
#include "vcs.h"
#include "vector.h"

#include <string>
#include <iostream>

/*
  Version 1.00 (03 Jul 2003)
  initial version - class that implements Message Digest MD5, as specified in
  Schneier, "Applied Cryptography", 1996, pp.436-441.
  Included comparison functions; added conversion to/from strings.

  Version 1.01 (04 Jul 2003)
  fixed bug in string() operator.

  Version 1.02 (04 Jul 2003)
  fixed bugs in Schneier's descriptions of MD5:
  * chaining variables should be initialised like SHA's
  * message length is low-order byte first
  * message is encoded into 32-bit words in low-order byte first

  Version 1.03 (05 Jul 2003)
  added information function to return message size

  Version 1.04 (5 Jul 2003)
  * added self-testing on creation of the first object.
  * fixed an obscure bug in the conversion from (signed) char to int32u

  Version 1.05 (17 Jul 2006)
  * in verify, first create an istringstream object, then pass that on to process, since
  this requires a pass by reference, which cannot be done by direct conversion.
  * in the constructor, made an explicit conversion of the output of floor to int32u.

  Version 1.06 (6 Oct 2006)
  modified for compatibility with VS .NET 2005:
  * in constructor, modified use of pow to avoid ambiguity

  Version 1.10 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libcomm {

class md5 {
   static const libbase::vcs version;
   // static variables
   static bool tested;
   // additive constants
   static libbase::vector<libbase::int32u> t;
   // rotational constants
   static const int s[];
   // message index constants
   static const int ndx[];
   // current hash value
   libbase::vector<libbase::int32u> m_hash;
   // size of message so far (used for termination)
   libbase::int64u m_size;
#ifdef _DEBUG
   // debugging variables
   bool m_padded, m_terminated;
#endif
public:
   // basic constructor/destructor
        md5();
        virtual ~md5();
   // conversion to/from strings
   md5(const std::string& s);
   operator std::string() const;
   // public interface for computing digest
        void reset();
   void process(const libbase::vector<libbase::int32u>& M);
   void process(const char *buf, const int size);
   void process(std::istream& sin);
   // information functions
   libbase::int64u size() const { return m_size; };
   // comparison functions
   bool operator>(const md5& x) const;
   bool operator<(const md5& x) const;
   bool operator==(const md5& x) const;
   bool operator!=(const md5& x) const;
protected:
   // verification function
   bool verify(const std::string message, const std::string hash);
   // circular shift
   static libbase::int32u cshift(const libbase::int32u x, const int s);
   // nonlinear functions
   static libbase::int32u f(const int i, const libbase::int32u X, const libbase::int32u Y, const libbase::int32u Z);
   // step operation
   static libbase::int32u op(const int i, const libbase::int32u a, const libbase::int32u b, const libbase::int32u c, const libbase::int32u d, const libbase::vector<libbase::int32u>& M);
   // stream input/output
   friend std::ostream& operator<<(std::ostream& sout, const md5& x);
   friend std::istream& operator>>(std::istream& sin, md5& x);
};

}; // end namespace

#endif
