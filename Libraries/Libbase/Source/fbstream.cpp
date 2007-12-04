#include "fbstream.h"
#include <iostream>

namespace libbase {

const vcs fbstream::version("Bitstream File-handling module (fbstream)", 1.10);
                    
void ofbstream::write_buffer()
   {
   // align first bit to write to lsb
   buffer >>= 32-ptr;
   // write as many bytes as necessary to expel buffer
   for(int i=0; i<ptr; i+=8)
      c.put(char(buffer.extract(i+7,i)));
   c << std::flush;
   ptr = 0;
   }

ofbstream::ofbstream(const char *name)
   {
   open(name);
   }

ofbstream::~ofbstream()
   {
   close();
   }

void ofbstream::open(const char *name)
   {
   using std::ios;
   c.open(name, ios::binary | ios::out | ios::trunc);
   ptr = 0;
   buffer = 0;
   }

void ofbstream::close()
   {
   if(ptr > 0)
      write_buffer();
   c.close();
   }

void ifbstream::read_buffer()
   {
   int ch = c.get();
   if(ch != EOF)
      {
      buffer = ch;
      ptr = 8;
      }
   }

ifbstream::ifbstream(const char *name)
   {
   open(name);
   }

ifbstream::~ifbstream()
   {
   close();
   }

void ifbstream::open(const char *name)
   {
   using std::ios;
   c.open(name, ios::binary | ios::in);
   ptr = 0;
   read_buffer();
   }

void ifbstream::close()
   {
   c.close();
   }

}; // end namespace
