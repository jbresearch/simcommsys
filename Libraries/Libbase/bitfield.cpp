/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "bitfield.h"

#include <cstdlib>
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

bitfield bitfield::operator|(const bitfield& x) const
   {
   bitfield y = *this;
   y |= x;
   return y;
   }

bitfield bitfield::operator&(const bitfield& x) const
   {
   bitfield y = *this;
   y &= x;
   return y;
   }

bitfield bitfield::operator^(const bitfield& x) const
   {
   bitfield y = *this;
   y ^= x;
   return y;
   }

bitfield bitfield::operator*(const bitfield& x) const
   {
   assertalways(this->bits == x.bits);
   bitfield y;
   y.bits = 1;
   int32u r = this->field & x.field;
   for (int i = 0; i < this->bits; i++)
      if (r & (1 << i))
         y.field ^= 1;
   return y;
   }

// Barrel-Shifting Operations

bitfield bitfield::operator+(const bitfield& x) const
   {
   bitfield y;
   y.bits = this->bits + x.bits;
   y.field = (this->field << x.bits) | x.field;
   return y;
   }

bitfield bitfield::operator<<(const bitfield& x) const
   {
   bitfield y;
   y.bits = this->bits;
   y.field = (this->field << x.bits) | x.field;
   y.field &= y.mask();
   return y;
   }

bitfield bitfield::operator>>(const bitfield& x) const
   {
   bitfield y;
   y.bits = x.bits;
   y.field = (this->field << (x.bits - this->bits)) | (x.field >> this->bits);
   y.field &= y.mask();
   return y;
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

bitfield bitfield::operator<<(const int x) const
   {
   bitfield c = *this;
   c <<= x;
   return c;
   }

bitfield bitfield::operator>>(const int x) const
   {
   bitfield c = *this;
   c >>= x;
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
