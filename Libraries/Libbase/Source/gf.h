#ifndef __gf_h
#define __gf_h

#include "config.h"
#include <iostream>
#include <string>

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
   - Defined conversions: to integer, to string.
   - Defined stream functions: output
   - Realizations: gf<8,283> Rijndael, gf<2>..gf<10> Lin & Costello
   - Added number of elements in the field as a static function
*/

template <int m, int poly> class gf {
public:
   /*! \name Class parameters */
   //! Number of elements in the field
   static int elements() { return 1<<m; };
   // @}

private:
   /*! \name Object representation */
   //! Representation of this element by its polynomial coefficients
   int32u value;
   // @}

public:
   /*! \name Constructors / Destructors */
   gf(int32u value=0);
   // @}

   /*! \name Type conversion */
   operator int32u() const { return value; };
   operator std::string() const;
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

/*! \name Stream Input/Output */
template <int m, int poly> std::ostream& operator<<(std::ostream& s, const gf<m,poly>& b);
//template <int m, int poly> std::istream& operator>>(std::istream& s, gf<m,poly>& b);
// @}

}; // end namespace

#endif
