#ifndef __sha_h
#define __sha_h

#include "config.h"
#include "vector.h"

#include <string>
#include <iostream>

namespace libcomm {

/*!
   \brief   Secure Hash Algorithm.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (19 Sep 2002)
   initial version - class that implements Secure Hash Algorithm, as specified in
   Schneier, "Applied Cryptography", 1996, pp.442-445.

   \version 1.01 (20 Sep 2002)
   added comparison functions; added conversion to/from strings.

   \version 1.02 (03 Jul 2003)
   cleaned up nonlinear function implementation - now everything is done in function
   'f', instead of calling one of four functions from there.

   \version 1.03 (04 Jul 2003)
   fixed bug in string() operator.

   \version 1.04 (5 Jul 2003)
   - fixed an obscure bug in the conversion from (signed) char to int32u

   \version 1.10 (6 Nov 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class sha {
   // additive constants
   static const libbase::int32u K[];
   // current hash value
   libbase::vector<libbase::int32u> m_hash;
   // size of message so far (used for termination)
   libbase::int64u m_size;
#ifndef NDEBUG
   // debugging variables
   bool m_padded, m_terminated;
#endif
public:
   // basic constructor/destructor
   sha();
   virtual ~sha() {};
   // conversion to/from strings
   sha(const std::string& s);
   operator std::string() const;
   // public interface for computing digest
   void reset();
   void process(const libbase::vector<libbase::int32u>& M);
   void process(const char *buf, const int size);
   void process(std::istream& sin);
   // comparison functions
   bool operator>(const sha& x) const;
   bool operator<(const sha& x) const;
   bool operator==(const sha& x) const;
   bool operator!=(const sha& x) const;
protected:
   // Nonlinear functions
   static libbase::int32u f(const int t, const libbase::int32u X, const libbase::int32u Y, const libbase::int32u Z);
   // Circular shift
   static libbase::int32u cshift(const libbase::int32u x, const int s);
   // Message expander
   static void expand(const libbase::vector<libbase::int32u>& M, libbase::vector<libbase::int32u>& W);
   // stream input/output
   friend std::ostream& operator<<(std::ostream& sout, const sha& x);
   friend std::istream& operator>>(std::istream& sin, sha& x);
};

}; // end namespace

#endif
