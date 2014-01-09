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

#include "bstream.h"

namespace libbase {

bstream::bstream()
   {
   buffer.init(32);
   }

obstream& obstream::operator<<(const bitfield& b)
   {
   bitfield pending = b;
   int left = b.size();

   while (left)
      {
      int cur = std::min(left, 32 - ptr);
      buffer = pending.extract(cur - 1, 0) >> buffer;
      pending >>= cur;
      ptr += cur;
      if (ptr == 32)
         write_buffer();
      left -= cur;
      }

   return *this;
   }

ibstream& ibstream::operator>>(bitfield& b)
   {
   int left = b.size();

   while (left)
      {
      if (ptr > 0)
         {
         int cur = std::min(ptr, left);
         b = buffer.extract(cur - 1, 0) >> b;
         buffer >>= cur;
         ptr -= cur;
         left -= cur;
         }
      if (ptr == 0)
         read_buffer();
      }

   return *this;
   }

} // end namespace
