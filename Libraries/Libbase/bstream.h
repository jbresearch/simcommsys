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

#ifndef __bstream_h
#define __bstream_h

#include "config.h"
#include "bitfield.h"

namespace libbase {

/*!
 * \brief   Bitstream Base.
 * \author  Johann Briffa
 *
 * \version 1.01 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.02 (15 Jun 2002)
 * - modified implementation file min() function usage to specify global one.
 * - added default constructor and virtual destructor for obstream/ibstream
 *
 * \version 1.10 (26 Oct 2006)
 * - defined class and associated data within "libbase" namespace.
 */

class bstream {
protected:
   bitfield buffer; // a 32-bit buffer for read/write operations
   int ptr; // points to the first unused/unread bit
public:
   bstream();
};

class obstream : public bstream {
protected:
   virtual void write_buffer() = 0;
public:
   obstream()
      {
      }
   virtual ~obstream()
      {
      }
   obstream& operator<<(const bitfield& b);
};

class ibstream : public bstream {
protected:
   virtual void read_buffer() = 0;
public:
   ibstream()
      {
      }
   virtual ~ibstream()
      {
      }
   ibstream& operator>>(bitfield& b);
};

} // end namespace

#endif
