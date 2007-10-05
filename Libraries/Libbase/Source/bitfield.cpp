#include "bitfield.h"

#include <stdlib.h>
#include <string>

namespace libbase {

using std::cerr;

const vcs bitfield::version("Bitfield module (bitfield)", 1.30);
              
// Static Members
                     
int bitfield::defsize = 0;

// Private functions

int32u bitfield::mask() const
   {
   return bits==32 ? 0xffffffff : ((1L << bits) - 1L);
   }

void bitfield::check_range(int32u f) const
   {
   if(f & ~mask())
      {
      cerr << "FATAL ERROR (bitfield): value outside bitfield range (" << f << ").\n";
      exit(1);
      }
   }

void bitfield::check_fieldsize(int b)
   {
   if(b<0 || b>32)
      {
      cerr << "FATAL ERROR (bitfield): invalid bitfield size (" << b << ").\n";
      exit(1);
      }
   }

void bitfield::init(const char *s)
   {
   bits = 0;
   field = 0;
   const char *p;
   for(p=s; *p=='1' || *p=='0'; p++)
      {
      field <<= 1;
      if(*p == '1')
         field |= 1;
      bits++;
      }
   if(*p != '\0')
      cerr << "ERROR (bitfield): initialization string (" << s << ") contains invalud characters.\n";
   }

// Conversion operations

bitfield::operator std::string() const
   {
   std::string sTemp;
   for(int i=bits-1; i>=0; i--)
      sTemp += '0' + ((field >> i) & 1);
   return sTemp;
   }

// Creation and Destruction
   
bitfield::bitfield()
   {
   bits = defsize;
   field = 0;
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
   if(hi >= bits || lo < 0 || lo > hi)
      {
      cerr << "FATAL ERROR (bitfield): invalid range for extraction (" << hi << ", " << lo << ").\n";
      exit(1);
      }
   c.bits = hi-lo+1;
   c.field = (field >> lo) & c.mask();
   return c;
   }

bitfield bitfield::extract(const int b) const
   {
   bitfield c;
   if(b >= bits || b < 0)
      {
      cerr << "FATAL ERROR (bitfield): invalid range for extraction (" << b << ").\n";
      exit(1);
      }
   c.bits = 1;
   c.field = (field >> b) & 1;
   return c;
   }

// Logic Operations

bitfield& bitfield::operator|=(const bitfield& x)
   {
   if(bits != x.bits)
      {
      cerr << "FATAL ERROR (bitfield): bitfield OR failed; parameters have unequal size\n";
      exit(1);
      }
   field |= x.field;
   return *this;
   }

bitfield& bitfield::operator&=(const bitfield& x)
   {
   if(bits != x.bits)
      {
      cerr << "FATAL ERROR (bitfield): bitfield AND failed; parameters have unequal size\n";
      exit(1);
      }
   field &= x.field;
   return *this;
   }

bitfield& bitfield::operator^=(const bitfield& x)
   {
   if(bits != x.bits)
      {
      cerr << "FATAL ERROR (bitfield): bitfield XOR failed; parameters have unequal size\n";
      exit(1);
      }
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
   if(a.bits != b.bits)
      {
      cerr << "FATAL ERROR (bitfield): collapse failed; parameters have unequal size\n";
      exit(1);
      }
   bitfield c;
   c.bits = 1;
   int32u x = a.field & b.field;
   for(int i=0; i<a.bits; i++)
       if(x & (1<<i))
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
   if(x >= 32)	// avoid a subtle bug in gcc (or the Pentium, not sure which...)
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
   s << std::string(b);
   return s;
   }

std::istream& operator>>(std::istream& s, bitfield& b)
   {
   std::string str;
   s >> str;
   b.init(str.c_str());
   return s;
   }

}; // end namespace
