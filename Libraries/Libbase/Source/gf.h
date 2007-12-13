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

   \version 1.00 (11-13 Dec 2007)
   - Initial version; implements extensions of the binary field: \f$ GF(2^n) \f$.
   - This is the first class where we're not using the vcs version-printing class.
   - Defined operations: addition and multiplication.
   - Defined conversions: to integer, to string, from integer, from string.
   - Defined stream functions: output, input.
   - Realizations: gf<8,283> Rijndael, gf<2>..gf<10> Lin & Costello
   - Added number of elements in the field as a static function
   - Created initialization routine to convert from integer
   - Moved class-specific documentation here
   - Moved stream I/O functions here
   - Moved string conversion routine from constructor to a new init function

   
   \param   m     Order of the binary field extension; that is, the field will be \f$ GF(2^m) \f$.
   \param   poly  Primitive polynomial used to define the field elements

   In integer representations of polynomials (e.g \c poly), higher-order bits in the integer
   represent higher-order powers of the polynomial representation. For example:
   \f[ x^6 + x^4 + x^2 + x^1 + 1 = \{ 01010111 \}_2 = \{ 57 \}_16 = \{ 87 \}_10 \f]

   \warning Due to the internal representation, this class is limited to \f$ GF(2^31) \f$.
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

   /*! \name Internal functions */
   void init(int32u value);
   void init(const char *s);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   gf(int32u value=0) { init(value); };
   gf(const char *s) { init(s); };
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

template <int m, int poly> std::ostream& operator<<(std::ostream& s, const gf<m,poly>& b)
   {
   s << std::string(b);
   return s;
   }

template <int m, int poly> std::istream& operator>>(std::istream& s, gf<m,poly>& b)
   {
   std::string str;
   s >> str;
   b = str.c_str();
   return s;
   }

// @}

}; // end namespace

#endif
