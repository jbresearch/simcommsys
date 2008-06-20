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

   Implements Secure Hash Algorithm, as specified in Schneier, "Applied
   Cryptography", 1996, pp.442-445.
*/

class sha {
   /*! \name Class-wide constants */
   static const libbase::int32u K[];         //!< Additive constants
   // @}
   /*! \name Internally-used objects */
   libbase::vector<libbase::int32u> m_hash;  //!< Current hash value
   libbase::int64u m_size;    //!< Size of message so far (used for termination)
#ifndef NDEBUG
   bool m_padded;
   bool m_terminated;
#endif
   // @}
protected:
   /*! \name Internal functions */
   // Nonlinear functions
   static libbase::int32u f(const int t, const libbase::int32u X, const libbase::int32u Y, const libbase::int32u Z);
   // Circular shift
   static libbase::int32u cshift(const libbase::int32u x, const int s);
   // Message expander
   static void expand(const libbase::vector<libbase::int32u>& M, libbase::vector<libbase::int32u>& W);
   // @}

   /*! \name Stream input/output */
   friend std::ostream& operator<<(std::ostream& sout, const sha& x);
   friend std::istream& operator>>(std::istream& sin, sha& x);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   sha();
   virtual ~sha() {};
   // @}

   /*! \name Conversion operations */
   sha(const std::string& s);
   operator std::string() const;
   // @}

   /*! \name Interface for computing digest */
   void reset();
   void process(const libbase::vector<libbase::int32u>& M);
   void process(const char *buf, const int size);
   void process(std::istream& sin);
   void process(std::string& s);
   // @}

   /*! \name Comparison functions */
   bool operator>(const sha& x) const;
   bool operator<(const sha& x) const;
   bool operator==(const sha& x) const;
   bool operator!=(const sha& x) const;
   // @}
};

}; // end namespace

#endif
