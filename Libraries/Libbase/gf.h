#ifndef __gf_h
#define __gf_h

#include "config.h"
#include <iostream>
#include <string>

namespace libbase {

/*!
 * \brief   Galois Field Element.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * Implements extensions of the binary field: \f$ GF(2^n) \f$.
 * 
 * Realizations:
 * - gf<8,283> Rijndael,
 * - gf<2>..gf<10> Lin & Costello
 * 
 * \param   m     Order of the binary field extension; that is, the field will
 * be \f$ GF(2^m) \f$.
 * \param   poly  Primitive polynomial used to define the field elements
 * 
 * In integer representations of polynomials (e.g \c poly), higher-order bits in
 * the integer represent higher-order powers of the polynomial representation.
 * For example:
 * \f[ x^6 + x^4 + x^2 + x^1 + 1 = \{ 01010111 \}_2 = \{ 57 \}_16 = \{ 87 \}_10 \f]
 * 
 * \warning Due to the internal representation, this class is limited to
 * \f$ GF(2^31) \f$.
 */

template <int m, int poly>
class gf {
public:
   /*! \name Class parameters */
   //! Number of elements in the field
   static int elements()
      {
      return 1 << m;
      }

   //! dimension of the field over GF(2)
   static int dimension()
      {
      return m;
      }
   // @}

private:
   /*! \name Object representation */
   //! Representation of this element by its polynomial coefficients
   int value;
   // @}

   /*! \name Internal functions */
   void init(int value);
   void init(const char *s);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   gf(int value = 0)
      {
      init(value);
      }
   gf(const char *s)
      {
      init(s);
      }
   // @}

   /*! \name Type conversion */
   operator int() const
      {
      return value;
      }
   operator std::string() const;
   // @}

   /*! \name Arithmetic operations */
   gf& operator+=(const gf& x);
   gf& operator-=(const gf& x);
   gf& operator*=(const gf& x);
   gf& operator/=(const gf& x);
   gf inverse() const;
   // @}

};

/*! \name Arithmetic operations */

template <int m, int poly>
gf<m, poly> operator+(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c += b;
   }

template <int m, int poly>
gf<m, poly> operator-(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c -= b;
   }

template <int m, int poly>
gf<m, poly> operator*(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c *= b;
   }

template <int m, int poly>
gf<m, poly> operator/(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c /= b;
   }

// @}

/*! \name Stream Input/Output */

template <int m, int poly>
std::ostream& operator<<(std::ostream& s, const gf<m, poly>& b)
   {
   s << std::string(b);
   return s;
   }

template <int m, int poly>
std::istream& operator>>(std::istream& s, gf<m, poly>& b)
   {
   std::string str;
   s >> str;
   b = str.c_str();
   return s;
   }

// @}

} // end namespace

#endif
