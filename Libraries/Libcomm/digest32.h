/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __digest32_h
#define __digest32_h

#include "config.h"
#include "vector.h"

#include <string>
#include <iostream>
#include <vector>

namespace libcomm {

/*!
 * \brief   Common 32-bit Digest Implementation.
 * \author  Johann Briffa
 *
 * Implements methods and data handling that are common for digests using
 * 32-bit integer arithmetic and 64-byte block sizes.
 */

class digest32 {
   /*! \name Internally-used objects */
   libbase::int64u m_size; //!< Size of message so far (used for termination)
   bool m_padded; //!< Flag indicating message padding has been applied
   bool m_terminated; //!< Flag indicating message size has been included
   // @}
protected:
   /*! \name Internally-used objects */
   libbase::vector<libbase::int32u> m_hash; //!< Current hash value
   bool lsbfirst; //!< Bytes are placed in least-significant byte positions first
   // @}
   /*! \name Digest-specific functions */
   virtual void derived_reset() = 0;
   virtual void process_block(const libbase::vector<libbase::int32u>& M) = 0;
   // @}
   /*! \name Internal functions */
   void reset();
   void process(const unsigned char *buf, int size);
   void flush()
      {
      process(NULL, 0);
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   digest32();
   virtual ~digest32()
      {
      }
   // @}

   /*! \name Conversion operations */
   digest32(const std::string& s);
   operator std::string() const;
   operator std::vector<unsigned char>() const;
   // @}

   /*! \name Interface for computing digest */
   void process(std::istream& sin);
   void process(const std::vector<unsigned char>& v);
   // @}

   /*! \name Comparison functions */
   bool operator==(const digest32& x) const;
   bool operator!=(const digest32& x) const
      {
      return !operator==(x);
      }
   // @}
};

/*! \name Stream output */
std::ostream& operator<<(std::ostream& sout, const digest32& x);
// @}

} // end namespace

#endif
