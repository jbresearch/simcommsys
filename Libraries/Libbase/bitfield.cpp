/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "bitfield.h"

#include <stdlib.h>
#include <string>

namespace libbase {

using std::cerr;

// Static Members

int bitfield::defsize = 0;

// Private functions

int32u bitfield::mask() const
   {
   return bits == 32 ? 0xffffffff : ((1L << bits) - 1L);
   }

void bitfield::check_range(int32u f) const
   {
   assertalways((f & ~mask()) == 0);
   }

void bitfield::check_fieldsize(int b)
   {
   assertalways(b>=0 && b<=32);
   }

void bitfield::init(const char *s)
   {
   bits = 0;
   field = 0;
   const char *p;
   for (p = s; *p == '1' || *p == '0'; p++)
      {
      field <<= 1;
      field |= (*p == '1');
      bits++;
      }
   // check there do not remain any invalid characters
   assertalways(*p == '\0');
   }

// Conversion operations


/*!
 * \brief Convert bitfield to a string representation
 */
std::string bitfield::asstring() const
   {
   std::string s;
   for (int i = bits - 1; i >= 0; i--)
      s += '0' + ((field >> i) & 1);
   return s;
   }

// Creation and Destruction

bitfield::bitfield()
   {
   bits = defsize;
   field = 0;
   }

/*!
 * \brief Constructor to directly convert an integer at a specified width
 */
bitfield::bitfield(const int32u field, const int bits)
   {
   bitfield::bits = bits;
   bitfield::field = field;
   }

/*!
 * \brief Constructor that converts a vector of bits
 * 
 * Bits are held in the vector as low-order first; this means that the first
 * (index 0) element in the vector is the right-most (or least-significant) bit.
 * This convention is consistent with the convention used for bit indexing
 * using the [] operator, and also with that for converting vectors to integer
 * representation in fsm.
 */
bitfield::bitfield(const vector<bool>& v)
   {
   bits = v.size();
   check_fieldsize(bits);
   field = 0;
   for (int i = 0; i < bits; i++)
      field |= v(i) << i;
   }

/*!
 * \brief Convert bitfield to a vector representation
 * \sa bitfield()
 */
vector<bool> bitfield::asvector() const
   {
   vector<bool> result(bits);
   for (int i = 0; i < bits; i++)
      result(i) = ((field >> i) & 1);
   return result;
   }

// Resizing Operations

void bitfield::resize(const int b)
   {
   check_fieldsize(b);
   bits = b;
   field &= mask();
   }

void bitfield::setdefsize(const int b)
   {
   check_fieldsize(b);
   defsize = b;
   }

// Assignment Operations

bitfield& bitfield::operator=(const bitfield& x)
   {
   bits = x.bits;
   field = x.field;
   return *this;
   }

bitfield& bitfield::operator=(const int32u x)
   {
   check_range(x);
   field = x;
   return *this;
   }

// Extraction Operations

bitfield bitfield::extract(const int hi, const int lo) const
   {
   bitfield c;
   assertalways(hi < bits && lo >= 0 && lo <= hi);
   c.bits = hi - lo + 1;
   c.field = (field >> lo) & c.mask();
   return c;
   }

bitfield bitfield::extract(const int b) const
   {
   bitfield c;
   assertalways(b < bits && b >= 0);
   c.bits = 1;
   c.field = (field >> b) & 1;
   return c;
   }

// Bit Reversal

bitfield bitfield::reverse() const
   {
   bitfield result(0, 0);
   for (int i = 0; i < bits; i++)
      result = result + extract(i);
   return result;
   }

// Logic Operations

bitfield& bitfield::operator|=(const bitfield& x)
   {
   assertalways(bits == x.bits);
   field |= x.field;
   return *this;
   }

bitfield& bitfield::operator&=(const bitfield& x)
   {
   assertalways(bits == x.bits);
   field &= x.field;
   return *this;
   }

bitfield& bitfield::operator^=(const bitfield& x)
   {
   assertalways(bits == x.bits);
   field ^= x.field;
   return *this;
   }

// Logic Operations - friends

bitfield operator|(const bitfield& a, const bitfield& b)
   {
   bitfield c = a;
   c |= b;
   return c;
   }

bitfield operator&(const bitfield& a, const bitfield& b)
   {
   bitfield c = a;
   c &= b;
   return c;
   }

bitfield operator^(const bitfield& a, const bitfield& b)
   {
   bitfield c = a;
   c ^= b;
   return c;
   }

bitfield operator*(const bitfield& a, const bitfield& b)
   {
   assertalways(a.bits == b.bits);
   bitfield c;
   c.bits = 1;
   int32u x = a.field & b.field;
   for (int i = 0; i < a.bits; i++)
      if (x & (1 << i))
         c.field ^= 1;
   return c;
   }

// Barrel-Shifting Operations

bitfield operator+(const bitfield& a, const bitfield& b)
   {
   bitfield c;
   c.bits = a.bits + b.bits;
   c.field = (a.field << b.bits) | b.field;
   return c;
   }

bitfield operator<<(const bitfield& a, const bitfield& b)
   {
   bitfield c;
   c.bits = a.bits;
   c.field = (a.field << b.bits) | b.field;
   c.field &= c.mask();
   return c;
   }

bitfield operator>>(const bitfield& a, const bitfield& b)
   {
   bitfield c;
   c.bits = b.bits;
   c.field = (a.field << (b.bits - a.bits)) | (b.field >> a.bits);
   c.field &= c.mask();
   return c;
   }

bitfield& bitfield::operator<<=(const int x)
   {
   field <<= x;
   field &= mask();
   return *this;
   }

bitfield& bitfield::operator>>=(const int x)
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

bitfield operator<<(const bitfield& a, const int b)
   {
   bitfield c = a;
   c <<= b;
   return c;
   }

bitfield operator>>(const bitfield& a, const int b)
   {
   bitfield c = a;
   c >>= b;
   return c;
   }

// Input/Output Operations

std::ostream& operator<<(std::ostream& s, const bitfield& b)
   {
   s << b.asstring();
   return s;
   }

std::istream& operator>>(std::istream& s, bitfield& b)
   {
   std::string str;
   s >> str;
   b.init(str.c_str());
   return s;
   }

}
// end namespace
