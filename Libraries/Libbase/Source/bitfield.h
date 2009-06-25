#ifndef __bitfield_h
#define __bitfield_h

#include "config.h"
#include "vector.h"

#include <iostream>
#include <string>

namespace libbase {

/*!
   \brief   Bitfield (register of a set size).
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

class bitfield {
   static int   defsize;   //!< default size
   // member variables
   int32u field;           //!< bit field value
   int    bits;            //!< number of bits
private:
   int32u mask() const;
   void check_range(int32u f) const;
   static void check_fieldsize(int b);
   void init(const char *s);
public:
   bitfield();
   bitfield(const char *s) { init(s); };
   bitfield(const int32u field, const int bits);
   explicit bitfield(const vector<bool>& v);

   // Type conversion to integer/string
   operator int32u() const { return field; };
   operator std::string() const;

   // Field size methods
   int size() const { return bits; };
   void resize(const int b);
   static void setdefsize(const int b);

   // Copy and Assignment
   bitfield& operator=(const bitfield& x);
   bitfield& operator=(const int32u x);

   // Partial extraction and indexed access
   bitfield extract(const int hi, const int lo) const;
   bitfield extract(const int b) const;
   bitfield operator[](const int b) const { return extract(b); };

   // Logical operators - OR, AND, XOR
   friend bitfield operator|(const bitfield& a, const bitfield& b);
   friend bitfield operator&(const bitfield& a, const bitfield& b);
   friend bitfield operator^(const bitfield& a, const bitfield& b);
   bitfield& operator|=(const bitfield& x);
   bitfield& operator&=(const bitfield& x);
   bitfield& operator^=(const bitfield& x);

   // Convolution operator
   friend bitfield operator*(const bitfield& a, const bitfield& b);
   // Concatenation operator
   friend bitfield operator+(const bitfield& a, const bitfield& b);
   // Shift-register operators - sequence shift-in
   friend bitfield operator<<(const bitfield& a, const bitfield& b);
   friend bitfield operator>>(const bitfield& a, const bitfield& b);
   // Shift-register operators - zero shift-in
   friend bitfield operator<<(const bitfield& a, const int b);
   friend bitfield operator>>(const bitfield& a, const int b);
   bitfield& operator<<=(const int x);
   bitfield& operator>>=(const int x);

   // Stream I/O
   friend std::ostream& operator<<(std::ostream& s, const bitfield& b);
   friend std::istream& operator>>(std::istream& s, bitfield& b);
};

}; // end namespace

#endif
