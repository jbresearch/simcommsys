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

#ifndef __image_h
#define __image_h

#include "config.h"
#include "matrix.h"
#include "vector.h"
#include "serializer.h"

#include <iostream>

namespace libimage {

/*!
 * \brief   Image Class.
 * \author  Johann Briffa
 *
 * This class encapsulates the data and functions for dealing with a single
 * image, potentially containing a number of channels. According to common
 * convention in image processing, the origin is at the top left, so that
 * row-major order gives the normal raster conversion.
 */

template <class T>
class image : public libbase::serializable {
private:
   //! Internal image representation
   libbase::vector<libbase::matrix<T> > m_data;
   T m_lo;
   T m_hi;
   int m_maxval;
protected:
   //! Returns true if pixel values are scaled to [0.0,1.0]
   static bool is_scaled()
      {
      return (typeid(T) == typeid(double) || typeid(T) == typeid(float));
      }
   //! Set internal pixel value limits according to type
   void set_limits()
      {
      if (is_scaled())
         {
         m_lo = T(0.0);
         m_hi = T(1.0);
         }
      else
         {
         m_lo = T(0);
         m_hi = T(m_maxval);
         }
      }
public:
   // Construction / destruction
   explicit image(int rows = 0, int cols = 0, int c = 0, int maxval = 255) :
      m_maxval(maxval)
      {
      resize(rows, cols, c);
      set_limits();
      }
   virtual ~image()
      {
      }
   // resizing
   void resize(int rows, int cols, int c)
      {
      m_data.init(c);
      for (int i = 0; i < c; i++)
         m_data(i).init(rows, cols);
      }

   /*! \name Information functions */
   //! Minimum valid pixel value for internal representation
   T lo() const
      {
      return m_lo;
      }
   //! Maximum valid pixel value for internal representation
   T hi() const
      {
      return m_hi;
      }
   //! Maximum pixel value for saved file
   int range() const
      {
      return m_maxval;
      }
   //! Number of channels (image planes)
   int channels() const
      {
      return m_data.size();
      }
   //! Image size in rows and columns
   libbase::size_type<libbase::matrix> size() const
      {
      if (channels() > 0)
         return m_data(0).size();
      return libbase::size_type<libbase::matrix>(0, 0);
      }
   // @}

   /*! \name Pixel access */
   //! Extract channel as a matrix of pixel values
   libbase::matrix<T> getchannel(int c) const
      {
      assert(c >= 0 && c < channels());
      return m_data(c);
      }
   //! Copy matrix of pixel values to channel
   void setchannel(int c, const libbase::matrix<T>& m)
      {
      assert(c >= 0 && c < channels());
      assert(m.size() == size());
      m_data(c) = m;
      }
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(image)
DECLARE_SERIALIZER(image)
};

} // end namespace

#endif
