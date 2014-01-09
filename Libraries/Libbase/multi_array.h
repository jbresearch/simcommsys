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

#ifndef __multi_array_h
#define __multi_array_h

#include "config.h"
#include <boost/multi_array.hpp>

namespace boost {

/*!
 * \brief   Assignable version of Boost MultiArray.
 * \author  Johann Briffa
 */

template <typename T, std::size_t NumDims>
class assignable_multi_array : public multi_array<T, NumDims> {
public:
   /*! \name Constructors / Destructors */
   assignable_multi_array() :
      multi_array<T, NumDims> ()
      {
      }
   explicit assignable_multi_array(const detail::multi_array::extent_gen<
         NumDims>& ranges) :
      multi_array<T, NumDims> (ranges)
      {
      }
   assignable_multi_array(const assignable_multi_array& x) :
            multi_array<T, NumDims> (
                  dynamic_cast<const multi_array<T, NumDims>&> (x))
      {
      }
   // @}
   /*! \name Assignment */
   assignable_multi_array& operator=(const assignable_multi_array& x)
      {
      if (!std::equal(this->shape(), this->shape() + this->num_dimensions(),
            x.shape()))
         {
         resize(x.extents());
         for (std::size_t i = 0; i < NumDims; i++)
            libbase::trace << "DEBUG (multi_array): resized dimension " << i
                  << " = " << this->shape()[i] << std::endl;
         }
      dynamic_cast<multi_array<T, NumDims>&> (*this)
            = dynamic_cast<const multi_array<T, NumDims>&> (x);
      return *this;
      }
   /*! \name Scalar value assignment */
   assignable_multi_array& operator=(const T& x)
      {
      std::fill(this->data(), this->data() + this->num_elements(), x);
      return *this;
      }
   // @}
   /*! \name Informative functions */
   /*! \brief Get array extents description
    * Returns an object describing the array extents, in a format suitable
    * for use with resize().
    */
   detail::multi_array::extent_gen<NumDims> extents() const
      {
      typedef multi_array_types::extent_range extent_range;
      detail::multi_array::extent_gen<NumDims> extents_list;
      for (std::size_t i = 0; i < NumDims; i++)
         {
         extents_list.ranges_[i] = extent_range(this->index_bases()[i],
               this->index_bases()[i] + this->shape()[i]);
         libbase::trace << "DEBUG (multi_array): read dimension " << i << " = "
               << this->shape()[i] << std::endl;
         }
      return extents_list;
      }
   // @}
};

/*! \name Stream input/output */
/*! \brief Output multi_array to stream
 * Outputs array data in a format similar to libbase::matrix. Note that
 * unlike matrix, the output for a multi_array is not meant as a serialized
 * version (since the extent ranges do not necessarily start from zero).
 */
template <typename T>
std::ostream& operator<<(std::ostream& sout,
      const assignable_multi_array<T, 2>& x)
   {
   typedef multi_array_types::size_type size_type;
   typedef multi_array_types::index index;
   // print header (array size)
   sout << x.shape()[0] << '\t' << x.shape()[1] << std::endl;
   // print data
   index idx_i = x.index_bases()[0];
   for (size_type i = 0; i < x.shape()[0]; i++)
      {
      index idx_j = x.index_bases()[1];
      sout << x[idx_i][idx_j];
      for (size_type j = 1; j < x.shape()[1]; j++)
         {
         idx_j++;
         sout << '\t' << x[idx_i][idx_j];
         }
      sout << std::endl;
      idx_i++;
      }
   return sout;
   }
// @}

} // end namespace

#endif
