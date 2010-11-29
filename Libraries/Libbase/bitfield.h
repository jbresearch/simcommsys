#ifndef __bitfield_h
#define __bitfield_h

#include "config.h"
#include "vector.h"

#include <iostream>
#include <string>

namespace libbase {

/*!
 * \brief   Bitfield (register of a set size).
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class bitfield {
private:
   int32u field; //!< bit field value
   int bits; //!< number of bits
private:
   // check that given field size is representable by class
   static void check_fieldsize(int b)
      {
      assertalways(b>=0 && b<=32);
      }
   // Get binary mask covering bitfield
   int32u mask() const
      {
      return bits == 32 ? 0xffffffff : ((1L << bits) - 1L);
      }
   // Initialize value from string
   void set_fromstring(const char *s);
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   explicit bitfield(const int bits = 0) :
      field(0), bits(bits)
      {
      }
   //! Constructor to directly convert an integer at a specified width
   bitfield(const int32u field, const int bits) :
      field(field), bits(bits)
      {
      }
   //! Constructor to directly convert a string representation
   explicit bitfield(const char *s)
      {
      set_fromstring(s);
      }
   //! Constructor to directly convert a vector representation
   explicit bitfield(const vector<bool>& v);
   // @}

   /*! \name Type conversion */
   operator int32u() const
      {
      return field;
      }
   std::string asstring() const;
   vector<bool> asvector() const;
   // @}

   // Field size methods
   int size() const
      {
      return bits;
      }
   void init(const int b)
      {
      check_fieldsize(b);
      bits = b;
      field &= mask();
      }

   // Copy and Assignment
   bitfield& operator=(const bitfield& x)
      {
      bits = x.bits;
      field = x.field;
      return *this;
      }
   bitfield& operator=(const int32u x)
      {
      assert((x & ~mask()) == 0);
      field = x;
      return *this;
      }

   // Partial extraction
   bitfield extract(const int hi, const int lo) const
      {
      bitfield c;
      assert(hi < bits && lo >= 0 && lo <= hi);
      c.bits = hi - lo + 1;
      c.field = (field >> lo) & c.mask();
      return c;
      }
   bitfield extract(const int b) const
      {
      return extract(b, b);
      }
   // Indexed access
   bool operator()(const int b) const
      {
      assert(b < bits && b >= 0);
      return (field >> b) & 1;
      }

   // Bit-reversal method
   bitfield reverse() const
      {
      bitfield result;
      for (int i = 0; i < bits; i++)
         result = result + extract(i);
      return result;
      }

   // Logical operators - OR, AND, XOR
   bitfield operator|(const bitfield& x) const
      {
      bitfield y = *this;
      y |= x;
      return y;
      }
   bitfield operator&(const bitfield& x) const
      {
      bitfield y = *this;
      y &= x;
      return y;
      }
   bitfield operator^(const bitfield& x) const
      {
      bitfield y = *this;
      y ^= x;
      return y;
      }
   bitfield& operator|=(const bitfield& x)
      {
      assert(bits == x.bits);
      field |= x.field;
      return *this;
      }
   bitfield& operator&=(const bitfield& x)
      {
      assert(bits == x.bits);
      field &= x.field;
      return *this;
      }
   bitfield& operator^=(const bitfield& x)
      {
      assert(bits == x.bits);
      field ^= x.field;
      return *this;
      }

   // Convolution operator
   bitfield operator*(const bitfield& x) const
      {
      assert(this->bits == x.bits);
      bitfield y;
      y.bits = 1;
      int32u r = this->field & x.field;
      for (int i = 0; i < this->bits; i++)
         if (r & (1 << i))
            y.field ^= 1;
      return y;
      }
   // Concatenation operator
   bitfield operator+(const bitfield& x) const
      {
      bitfield y;
      y.bits = this->bits + x.bits;
      y.field = (this->field << x.bits) | x.field;
      return y;
      }
   // Shift-register operators - sequence shift-in
   bitfield operator<<(const bitfield& x) const
      {
      bitfield y;
      y.bits = this->bits;
      y.field = (this->field << x.bits) | x.field;
      y.field &= y.mask();
      return y;
      }
   bitfield operator>>(const bitfield& x) const
      {
      bitfield y;
      y.bits = x.bits;
      y.field = (this->field << (x.bits - this->bits))
            | (x.field >> this->bits);
      y.field &= y.mask();
      return y;
      }
   // Shift-register operators - zero shift-in
   bitfield operator<<(const int x) const
      {
      bitfield c = *this;
      c <<= x;
      return c;
      }
   bitfield operator>>(const int x) const
      {
      bitfield c = *this;
      c >>= x;
      return c;
      }
   bitfield& operator<<=(const int x)
      {
      field <<= x;
      field &= mask();
      return *this;
      }
   bitfield& operator>>=(const int x)
      {
      if (x >= 32) // avoid a subtle bug in gcc (or the Pentium, not sure which...)
         field = 0;
      else
         {
         field >>= x;
         field &= mask();
         }
      return *this;
      }

   // Stream I/O
   friend std::ostream& operator<<(std::ostream& s, const bitfield& b)
      {
      s << b.asstring();
      return s;
      }
   friend std::istream& operator>>(std::istream& s, bitfield& b)
      {
      std::string str;
      s >> str;
      b.set_fromstring(str.c_str());
      return s;
      }
};

} // end namespace

#endif
