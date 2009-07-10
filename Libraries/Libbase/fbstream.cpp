/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "fbstream.h"
#include <iostream>

namespace libbase {

void ofbstream::write_buffer()
   {
   // align first bit to write to lsb
   buffer >>= 32 - ptr;
   // write as many bytes as necessary to expel buffer
   for (int i = 0; i < ptr; i += 8)
      c.put(char(buffer.extract(i + 7, i)));
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
   c.open(name, std::ios_base::out | std::ios_base::trunc
         | std::ios_base::binary);
   ptr = 0;
   buffer = 0;
   }

void ofbstream::close()
   {
   if (ptr > 0)
      write_buffer();
   c.close();
   }

void ifbstream::read_buffer()
   {
   int ch = c.get();
   if (ch != EOF)
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
   c.open(name, std::ios_base::in | std::ios_base::binary);
   ptr = 0;
   read_buffer();
   }

void ifbstream::close()
   {
   c.close();
   }

} // end namespace
