/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "gf.h"

#include <stdlib.h>
#include <string>

namespace libbase {

using std::cerr;


// Internal functions

/*!
   \brief Initialization
   \param   value Representation of element by its polynomial coefficients

   \todo Validate \c poly - this should be a primitive polynomial [cf. Lin & Costello, 2004, p.41]
*/
template <int m, int poly> void gf<m,poly>::init(int32u value)
   {
   assert(m < 32);
   assert(value < (1<<m));
   gf::value = value;
   }

/*!
   \brief Conversion from string
   \param   s     String representation of element by its polynomial coefficients (binary)

   This function converts the string to an integer and calls init().
   The string must only contain 1's and 0's.
*/
template <int m, int poly> void gf<m,poly>::init(const char *s)
   {
   int32u value = 0;
   const char *p;
   for(p=s; *p=='1' || *p=='0'; p++)
      {
      value <<= 1;
      if(*p == '1')
         value |= 1;
      }
   assert(*p == '\0');
   init(value);
   }


// Conversion operations

template <int m, int poly> gf<m,poly>::operator std::string() const
   {
   std::string sTemp;
   for(int i=m-1; i>=0; i--)
      sTemp += '0' + ((value >> i) & 1);
   return sTemp;
   }


// Arithmetic operations

/*!
   \brief Addition
   \param   x  Field element we want to add to this one.

   Addition within extensions of a field is the addition of the corresponding coefficients
   in the polynomial representation. When the field characteristic is 2 (ie. for extensions
   of a binary field), addition of the coefficients is equivalent to an XOR operation.
*/
template <int m, int poly> gf<m,poly>& gf<m,poly>::operator+=(const gf<m,poly>& x)
   {
   value ^= x.value;
   return *this;
   }

/*!
   \brief Multiplication
   \param   x  Field element we want to multiply to this one (ie. multiplicand).

   Multiplication within extensions of a field is the multiplication of the polynomials
   representing the two values. This can be done by the usual long-multiplication
   algorithm. Every time the result overflows, we need to subtract the modular polynomial;
   for extensions of a binary field, this is achieved by an XOR operation.

   [cf. Gladman, "A Specification for Rijndael, the AES Algorithm", 2003, pp.3-4]
*/
template <int m, int poly> gf<m,poly>& gf<m,poly>::operator*=(const gf<m,poly>& x)
   {
   // Copy the multiplier (A) and multiplicand (B)
   int32u A = value;
   int32u B = x.value;
   // Initialize result
   value = 0;
   // Loop over all bits in multiplicand
   for(int i=0; i<m && B!=0; i++)
      {
      // If the corresponding bit in the multiplicand is set,
      // add (XOR) the shifted multiplier
      if(B & 1)
         value ^= A;
      // Shift the multiplicand
      B >>= 1;
      // Shift the multiplier, subtracting the polynomial on overflow
      A <<= 1;
      if(A & (1<<m))
         A ^= poly;
      }
   return *this;
   }


// Explicit Realizations

// cf. Lin & Costello, 2004, App. A

template class gf<2,0x7>;     // 1 { 11 }
template class gf<3,0xB>;     // 1 { 011 }
template class gf<4,0x13>;    // 1 { 0011 }
template class gf<5,0x25>;    // 1 { 0 0101 }
template class gf<6,0x43>;    // 1 { 00 0011 }
template class gf<7,0x89>;    // 1 { 000 1001 }
template class gf<8,0x11D>;   // 1 { 0001 1101 }
template class gf<9,0x211>;   // 1 { 0 0001 0001 }
template class gf<10,0x409>;   // 1 { 00 0000 1001 }

// Rijndael field cf. Gladman, 2003, p.5

template class gf<8,0x11B>;   // 1 { 0001 1011 }

}; // end namespace
