#ifndef __md5_h
#define __md5_h

#include "config.h"
#include "vector.h"

#include <string>
#include <iostream>

namespace libcomm {

/*!
   \brief   Message Digest MD5 Algorithm.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements Message Digest MD5, as specified in Schneier, "Applied
   Cryptography", 1996, pp.436-441.
   Class performs self-testing on creation of the first object.

   \note there are bugs in Schneier's descriptions of MD5:
         - chaining variables should be initialised like SHA's
         - message length is low-order byte first
         - message is encoded into 32-bit words in low-order byte first
*/

class md5 {
   /*! \name Class-wide constants */
   static bool tested;        //!< Flag to indicate self-test has been done
   static libbase::vector<libbase::int32u> t;   //!< Additive constants
   static const int s[];      //!< Rotational constants
   static const int ndx[];    //!< Message index constants
   // @}
   /*! \name Internally-used objects */
   libbase::vector<libbase::int32u> m_hash;     //!< Current hash value
   libbase::int64u m_size;    //!< Size of message so far (used for termination)
#ifndef NDEBUG
   bool m_padded;
   bool m_terminated;
#endif
   // @}
protected:
   /*! \name Internal functions */
   // verification function
   bool verify(const std::string message, const std::string hash);
   // circular shift
   static libbase::int32u cshift(const libbase::int32u x, const int s);
   // nonlinear functions
   static libbase::int32u f(const int i, const libbase::int32u X, const libbase::int32u Y, const libbase::int32u Z);
   // step operation
   static libbase::int32u op(const int i, const libbase::int32u a, const libbase::int32u b, const libbase::int32u c, const libbase::int32u d, const libbase::vector<libbase::int32u>& M);
   // @}

   /*! \name Stream input/output */
   friend std::ostream& operator<<(std::ostream& sout, const md5& x);
   friend std::istream& operator>>(std::istream& sin, md5& x);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   md5();
   virtual ~md5() {};
   // @}

   /*! \name Conversion operations */
   md5(const std::string& s);
   operator std::string() const;
   // @}

   /*! \name Interface for computing digest */
   void reset();
   void process(const libbase::vector<libbase::int32u>& M);
   void process(const char *buf, const int size);
   void process(std::istream& sin);
   // @}

   /*! \name Information functions */
   libbase::int64u size() const { return m_size; };
   // @}

   /*! \name Comparison functions */
   bool operator>(const md5& x) const;
   bool operator<(const md5& x) const;
   bool operator==(const md5& x) const;
   bool operator!=(const md5& x) const;
   // @}
};

}; // end namespace

#endif
