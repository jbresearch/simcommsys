/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
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
