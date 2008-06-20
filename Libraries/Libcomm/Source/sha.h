#ifndef __sha_h
#define __sha_h

#include "config.h"
#include "vector.h"
#include "digest32.h"

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

class sha : public digest32 {
   /*! \name Class-wide constants */
   static const libbase::int32u K[];         //!< Additive constants
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
   /*! \name Digest-specific functions */
   void derived_reset();
   void process_block(const libbase::vector<libbase::int32u>& M);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   sha();
   // @}
};

}; // end namespace

#endif
