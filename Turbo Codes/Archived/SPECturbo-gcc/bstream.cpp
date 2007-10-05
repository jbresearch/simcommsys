#include "bstream.h"

const vcs bstream_version("Bitstream Base module (bstream)", 1.00);

bstream::bstream()
   {
   buffer.resize(32);
   }

obstream& obstream::operator<<(const bitfield& b)
   {
   bitfield pending = b;
   int left = b.size();

   while(left)
      {
      int cur = min(left, 32-ptr);
      buffer = pending.extract(cur-1, 0) >> buffer;
      pending >>= cur;
      ptr += cur;
      if(ptr == 32)
         write_buffer();
      left -= cur;
      }
   
   return *this;
   }

ibstream& ibstream::operator>>(bitfield& b)
   {
   int left = b.size();
   
   while(left)
      {
      if(ptr > 0)
         {
         int cur = min(ptr, left);
         b = buffer.extract(cur-1, 0) >> b;
         buffer >>= cur;
         ptr -= cur;
         left -= cur;
         }
      if(ptr == 0)
         read_buffer();
      }

   return *this;
   }
