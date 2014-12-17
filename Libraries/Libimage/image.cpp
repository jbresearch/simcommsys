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

#include "image.h"

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

namespace libimage {

// Saving/loading functions

template <class T>
std::ostream& image<T>::serialize(std::ostream& sout) const
   {
   libbase::trace << "Saving image" << std::flush;
   // header data
   const int chan = channels();
   assert(chan > 0);
   const int rows = m_data(0).size().rows();
   const int cols = m_data(0).size().cols();
   libbase::trace << " (" << cols << "×" << rows << "×" << chan << ")..."
         << std::flush;
   // write file descriptor
   if (chan == 1 && m_maxval == 1)
      sout << "P4" << std::endl; // bitmap
   else if (chan == 1 && m_maxval > 1)
      sout << "P5" << std::endl; // graymap
   else if (chan == 3)
      sout << "P6" << std::endl; // pixmap
   else
      failwith("Image format not supported");
   // write comment
   sout << "# file written by libimage" << std::endl;
   // write image size
   sout << cols << " " << rows << std::endl;
   // if needed, write maxval
   if (chan > 1 || m_maxval > 1)
      sout << m_maxval << std::endl;
   // write image data
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         for (int c = 0; c < chan; c++)
            {
            int p;
            if (typeid(T) == typeid(double) || typeid(T) == typeid(float))
               p = int(round(m_data(c)(i, j) * m_maxval));
            else
               p = int(m_data(c)(i, j));
            assert(p >= 0 && p <= m_maxval);
            if (m_maxval > 255) // 16-bit binary files (MSB first)
               {
               sout.put(p >> 8);
               p &= 0xff;
               }
            sout.put(p);
            }
   // done
   libbase::trace << "done" << std::endl;
   return sout;
   }

template <class T>
std::istream& image<T>::serialize(std::istream& sin)
   {
   libbase::trace << "Loading image" << std::flush;
   // header data
   int cols, rows, chan;
   bool binary;
   // read file header
   std::string line;
   std::getline(sin, line);
   // read file descriptor
   int descriptor;
   assert(line[0] == 'P');
   std::istringstream(line.substr(1)) >> descriptor;
   assertalways(descriptor >= 1 && descriptor <= 6);
   // determine the number of channels
   if (descriptor == 3 || descriptor == 6)
      chan = 3;
   else
      chan = 1;
   // determine the data format
   if (descriptor >= 4 || descriptor <= 6)
      binary = true;
   else
      binary = false;
   // skip comments or empty lines
   do
      {
      std::getline(sin, line);
      } while (line.size() == 0 || line[0] == '#');
   // read image size
   std::istringstream(line) >> cols >> rows;
   // if necessary read pixel value range
   if (descriptor == 1 || descriptor == 4)
      {
      m_maxval = 1;
      assertalways(!binary); // cannot handle binary bitmaps (packed bits)
      }
   else
      {
      std::getline(sin, line);
      std::istringstream(line) >> m_maxval;
      }
   libbase::trace << " (" << cols << "×" << rows << "×" << chan << ")...";
   // set interal representation limits
   set_limits();
   // set up space to hold image
   m_data.init(chan);
   for (int c = 0; c < chan; c++)
      m_data(c).init(rows, cols);
   // read image data
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         for (int c = 0; c < chan; c++)
            {
            if (binary)
               {
               int p = sin.get();
               if (m_maxval > 255) // 16-bit binary files (MSB first)
                  p = (p << 8) + sin.get();
               m_data(c)(i, j) = T(p);
               }
            else
               sin >> m_data(c)(i, j);
            assert(m_data(c)(i, j) >= 0 && m_data(c)(i, j) <= m_maxval);
            }
   assertalways(sin);
   // scale down if we're using floating-point
   if (is_scaled())
      for (int c = 0; c < chan; c++)
         m_data(c) /= T(m_maxval);
   // done
   libbase::trace << "done" << std::endl;
   return sin;
   }

} // end namespace

namespace libimage {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define SYMBOL_TYPE_SEQ \
   (int)(float)(double)

/* Serialization string: image<type>
 * where:
 *      type = int | float | double
 */
#define INSTANTIATE(r, x, type) \
      template class image<type>; \
      template <> \
      const serializer image<type>::shelper( \
            "image", \
            "image<" BOOST_PP_STRINGIZE(type) ">", \
            image<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // end namespace
