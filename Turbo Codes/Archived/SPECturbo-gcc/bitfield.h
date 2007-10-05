#ifndef __bitfield_h
#define __bitfield_h

#include "config.h"
#include "vcs.h"
#include <iostream.h>
#include <stdlib.h>

extern const vcs bitfield_version;

class bitfield {
   static int	defsize;	// default size
   int32u		field;	// bit field
   int			bits;		// number of bits
   int32u mask() const;
   void check_fieldsize(int b);
   void check_range(int32u f);
public:
   bitfield();
   bitfield(const char *s);
   operator int32u() const;
   
   int size() const;
   void resize(const int b);
   void setdefsize(const int b);
   
   bitfield& operator=(const bitfield& x);
   bitfield& operator=(const int32u x);

   bitfield extract(const int hi, const int lo);
   bitfield extract(const int b);
   bitfield operator[](const int b);
   
   friend bitfield operator|(const bitfield& a, const bitfield& b);
   friend bitfield operator&(const bitfield& a, const bitfield& b);
   friend bitfield operator^(const bitfield& a, const bitfield& b);
   friend bitfield operator*(const bitfield& a, const bitfield& b);
   bitfield& operator|=(const bitfield& x);
   bitfield& operator&=(const bitfield& x);
   bitfield& operator^=(const bitfield& x);
   
   friend bitfield operator,(const bitfield& a, const bitfield &b);
   friend bitfield operator<<(const bitfield& a, const bitfield& b);
   friend bitfield operator>>(const bitfield& a, const bitfield& b);
   bitfield& operator<<=(const int x);
   bitfield& operator>>=(const int x);
   friend bitfield operator<<(const bitfield& a, const int b);
   friend bitfield operator>>(const bitfield& a, const int b);

   friend ostream& operator<<(ostream& s, const bitfield& b);
};

// Private functions

inline int32u bitfield::mask() const
   {
   return bits==32 ? 0xffffffff : ((1L << bits) - 1L);
   }

inline void bitfield::check_fieldsize(int b)
   {
   if(b<0 || b>32)
      {
      cerr << "FATAL ERROR (bitfield): invalid bitfield size (" << b << ").\n";
      exit(1);
      }
   }
   
inline void bitfield::check_range(int32u f)
   {
   if(f & ~mask())
      {
      cerr << "FATAL ERROR (bitfield): value outside bitfield range (" << f << ").\n";
      exit(1);
      }
   }

// Creation and Destruction
   
inline bitfield::bitfield()
   {
   bits = defsize;
   field = 0;
   }

// Automatic Conversion

inline bitfield::operator int32u() const
   {
   return field;
   }

// Resizing Operations
   
inline int bitfield::size() const
   {
   return bits;
   }

inline void bitfield::resize(const int b)
   {
   check_fieldsize(b);
   bits = b;
   field &= mask();
   }

inline void bitfield::setdefsize(const int b)
   {
   check_fieldsize(b);
   defsize = b;
   }

// Assignment Operations

inline bitfield& bitfield::operator=(const bitfield& x)
   {
   bits = x.bits;
   field = x.field;
   return *this;
   }
   
inline bitfield& bitfield::operator=(const int32u x)
   {
   check_range(x);
   field = x;
   return *this;
   }

// Extraction Operations

inline bitfield bitfield::extract(const int hi, const int lo)
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

inline bitfield bitfield::extract(const int b)
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

inline bitfield bitfield::operator[](const int b)
   {
   return extract(b);
   }

// Logic Operations

inline bitfield& bitfield::operator|=(const bitfield& x)
   {
   if(bits != x.bits)
      {
      cerr << "FATAL ERROR (bitfield): bitfield OR failed; parameters have unequal size\n";
      exit(1);
      }
   field |= x.field;
   return *this;
   }

inline bitfield& bitfield::operator&=(const bitfield& x)
   {
   if(bits != x.bits)
      {
      cerr << "FATAL ERROR (bitfield): bitfield AND failed; parameters have unequal size\n";
      exit(1);
      }
   field &= x.field;
   return *this;
   }

inline bitfield& bitfield::operator^=(const bitfield& x)
   {
   if(bits != x.bits)
      {
      cerr << "FATAL ERROR (bitfield): bitfield XOR failed; parameters have unequal size\n";
      exit(1);
      }
   field ^= x.field;
   return *this;
   }

// Barrel-Shifting Operations

inline bitfield operator,(const bitfield& a, const bitfield &b)
   {
   bitfield c;
   c.bits = a.bits + b.bits;
   c.field = (a.field << b.bits) | b.field;
   return c;
   }

inline bitfield operator<<(const bitfield& a, const bitfield& b)
   {
   bitfield c;
   c.bits = a.bits;
   c.field = (a.field << b.bits) | b.field;
   c.field &= c.mask();
   return c;
   }

inline bitfield operator>>(const bitfield& a, const bitfield& b)
   {
   bitfield c;
   c.bits = b.bits;
   c.field = (a.field << (b.bits - a.bits)) | (b.field >> a.bits);
   c.field &= c.mask();
   return c;
   }

inline bitfield& bitfield::operator<<=(const int x)
   {
   field <<= x;
   field &= mask();
   return *this;
   }

inline bitfield& bitfield::operator>>=(const int x)
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

inline bitfield operator<<(const bitfield& a, const int b)
   {
   bitfield c = a;
   c <<= b;
   return c;
   }

inline bitfield operator>>(const bitfield& a, const int b)
   {
   bitfield c = a;
   c >>= b;
   return c;
   }

// Input/Output Operations

inline ostream& operator<<(ostream& s, const bitfield& b)
   {        
   for(int i=b.bits-1; i>=0; i--)
      s << ((b.field >> i) & 1);
   return s;
   }
   
#endif
