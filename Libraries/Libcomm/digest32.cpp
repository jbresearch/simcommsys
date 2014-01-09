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

#include "digest32.h"

#include <sstream>

namespace libcomm {

// Construction/Destruction

digest32::digest32()
   {
   // reset size counter
   m_size = 0;
   // reset flags
   m_padded = false;
   m_terminated = false;
   }

// Internal functions

void digest32::reset()
   {
   // reset size counter
   m_size = 0;
   // reset flags
   m_padded = false;
   m_terminated = false;
   // reset derived class
   derived_reset();
   }

/*!
 * \brief Converts a 64-byte block and passes for processing
 *
 * Depending on flag, bytes in the input buffer are placed in least-
 * significant byte positions of the 16x32-bit block first (ie. message
 * block is little-endian).
 *
 * \note If there are less than 64 bytes, padding is applied
 *
 * \note If after padding there is enough space left, message length is
 * included.
 */
void digest32::process(const unsigned char *buf, int size)
   {
   assert(size <= 64);
   if (m_padded && m_terminated)
      return;
   // convert message block and process
   libbase::vector<libbase::int32u> M(16);
   M = 0;
   for (int i = 0; i < size; i++)
      M(i >> 2) |= libbase::int8u(buf[i]) << 8 * (lsbfirst ? (i & 3) : 3 - (i
            & 3));
   // add padding (1-bit followed by zeros) if it fits and is necessary
   if (size < 64 && !m_padded)
      {
      M(size >> 2) |= libbase::int8u(0x80) << 8 * (lsbfirst ? (size & 3) : 3
            - (size & 3));
      m_padded = true;
      }
   // update size counter
   m_size += size;
   // add file size (in bits) if it fits and is necessary
   // (note that we need to fit the 8-byte size AND 1 byte of padding)
   if (size < 64 - 8 && !m_terminated)
      {
      assert((m_size >> 61) == 0);
      M(lsbfirst ? 14 : 15) = libbase::int32u(m_size << 3);
      M(lsbfirst ? 15 : 14) = libbase::int32u(m_size >> 29);
      m_terminated = true;
      }
   // go through the digest algorithm
   process_block(M);
   }

// Conversion to/from strings

digest32::digest32(const std::string& s)
   {
   // reset size counter
   m_size = 0;
   // reset flags
   m_padded = true;
   m_terminated = true;
   // load from std::string
   std::istringstream sin(s);
   // skip initial whitespace
   sin >> libbase::eatcomments >> std::ws >> libbase::verify;
   for (int i = 0; i < m_hash.size(); i++)
      {
      char buf[9];
      buf[8] = 0;
      assert(sin.read(buf, 8));
      m_hash(i) = strtoul(buf, NULL, 16);
      }
   }

digest32::operator std::string() const
   {
   assert(m_padded);
   assert(m_terminated);
   // write into a std::string
   std::ostringstream sout;
   for (int i = 0; i < m_hash.size(); i++)
      {
      sout.width(8);
      sout.fill('0');
      sout << std::hex << m_hash(i);
      }
   return sout.str();
   }

digest32::operator std::vector<unsigned char>() const
   {
   assert(m_padded);
   assert(m_terminated);
   // write into a std::vector of bytes, msb first
   std::vector<unsigned char> v(m_hash.size() * 4);
   for (int i = 0; i < m_hash.size(); i++)
      for (int j = 0; j < 4; j++)
         {
         v[i * 4 + j] = (m_hash(i) >> 8 * (3 - j)) & 0xff;
         }
   return v;
   }

// Interface for computing digest

void digest32::process(std::istream& sin)
   {
   // initialize the variables
   reset();
   // process whole data stream
   while (!sin.eof())
      {
      char buf[64];
      sin.read(buf, 64);
      process((unsigned char *)buf, int(sin.gcount()));
      }
   // flush to include stream length if necessary
   flush();
   }

void digest32::process(const std::vector<unsigned char>& v)
   {
   // initialize the variables
   reset();
   // process whole data array
   int at = 0;
   int left = v.size();
   while (left > 0)
      {
      // determine current block size and process
      const int cur = std::min(left, 64);
      process(&v[at], cur);
      // update array pointers
      at += cur;
      left -= cur;
      }
   // flush to include stream length if necessary
   flush();
   }

// Comparison functions

bool digest32::operator==(const digest32& x) const
   {
   assert(m_hash.size() == x.m_hash.size());
   for (int i = 0; i < m_hash.size(); i++)
      if (m_hash(i) != x.m_hash(i))
         return false;
   return true;
   }

// Stream input/output

std::ostream& operator<<(std::ostream& sout, const digest32& x)
   {
   return sout << std::string(x);
   }

} // end namespace
