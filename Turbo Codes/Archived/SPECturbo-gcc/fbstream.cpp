#include "fbstream.h"
#include <iostream.h>

const vcs fbstream_version("Bitstream File-handling module (fbstream)", 1.00);
                    
void ofbstream::write_buffer()
   {
   c << char(buffer.extract(7,0));
   c << char(buffer.extract(15,8));
   c << char(buffer.extract(23,16));
   c << char(buffer.extract(31,24));
   c << flush;
   ptr = 0;
   }

ofbstream::ofbstream(char name[])
   {
   open(name);
   }

ofbstream::~ofbstream()
   {
   close();
   }

void ofbstream::open(char name[])
   {
   c.open(name);
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
   unsigned char ch;
   bitfield b;
   b.resize(8);
   c >> ch; b = ch; buffer = b >> buffer;
   c >> ch; b = ch; buffer = b >> buffer;
   c >> ch; b = ch; buffer = b >> buffer;
   c >> ch; b = ch; buffer = b >> buffer;
   ptr = 32;
   }

ifbstream::ifbstream(char name[])
   {
   open(name);
   }

ifbstream::~ifbstream()
   {
   close();
   }

void ifbstream::open(char name[])
   {
   c.open(name);
   ptr = 0;
   read_buffer();
   }

void ifbstream::close()
   {
   c.close();
   }
