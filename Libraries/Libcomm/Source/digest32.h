#ifndef __digest32_h
#define __digest32_h

#include "config.h"
#include "vector.h"

#include <string>
#include <iostream>

namespace libcomm {

/*!
   \brief   Common 32-bit Digest Implementation.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements methods and data handling that are common for digests using
   32-bit integer arithmetic and 64-byte block sizes.
*/

class digest32 {
   /*! \name Internally-used objects */
   libbase::int64u m_size;    //!< Size of message so far (used for termination)
   bool m_padded;             //!< Flag indicating message padding has been applied
   bool m_terminated;         //!< Flag indicating message size has been included
   // @}
protected:
   /*! \name Internally-used objects */
   libbase::vector<libbase::int32u> m_hash;  //!< Current hash value
   // @}
   /*! \name Digest-specific functions */
   virtual void derived_reset() = 0;
   virtual void process_block(const libbase::vector<libbase::int32u>& M) = 0;
   // @}
   /*! \name Internal functions */
   void reset();
   void process(const char *buf, int size);
   void flush() { process(NULL, 0); };
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   digest32();
   virtual ~digest32() {};
   // @}

   /*! \name Conversion operations */
   digest32(const std::string& s);
   operator std::string() const;
   // @}

   /*! \name Interface for computing digest */
   void process(std::istream& sin);
   // @}

   /*! \name Comparison functions */
   bool operator==(const digest32& x) const;
   bool operator!=(const digest32& x) const { return !operator==(x); };
   // @}
};

/*! \name Stream output */
std::ostream& operator<<(std::ostream& sout, const digest32& x);
// @}

}; // end namespace

#endif
