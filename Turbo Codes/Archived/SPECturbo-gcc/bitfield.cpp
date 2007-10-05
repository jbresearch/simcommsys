#include "bitfield.h"

const vcs bitfield_version("Bitfield module (bitfield)", 1.00);
                     
// Static Members
                     
int bitfield::defsize = 0;

// Conversion Operations

bitfield::bitfield(const char *s)
   {
   bits = 0;
   field = 0;
   for(const char *p=s; *p=='1' || *p=='0'; p++)
      {
      field <<= 1;
      if(*p == '1')
         field |= 1;
      bits++;
      }
   }

// Logic Operations
   
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
