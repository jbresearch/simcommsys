#ifndef __gf_h
#define __gf_h

#include "config.h"

namespace libbase {

/*!
   \brief   Galois Field Element.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (11-12 Dec 2007)
   - Initial version; implements extensions of the binary field: \f$ GF(2^n) \f$.
   - This is the first class where we're not using the vcs version-printing class.
   - Defined operations: addition and multiplication.
*/

template <int m, int poly> class gf {
private:
   /*! \name Object representation */
   //! Representation of this element by its polynomial coefficients
   int32u value;
   // @}

public:
   /*! \name Constructors / Destructors */
   gf(int32u value=0);
   // @}

   /*! \name Arithmetic operations */
   gf& operator+=(const gf& x);
   gf& operator*=(const gf& x);
   // @}
};

/*! \name Arithmetic operations */
template <int m, int poly> gf<m,poly> operator+(const gf<m,poly>& a, const gf<m,poly>& b);
template <int m, int poly> gf<m,poly> operator*(const gf<m,poly>& a, const gf<m,poly>& b);
// @}

}; // end namespace

#endif
