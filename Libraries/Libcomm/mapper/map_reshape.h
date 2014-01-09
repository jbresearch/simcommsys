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

#ifndef __map_reshape_h
#define __map_reshape_h

#include "mapper.h"

namespace libcomm {

/*!
 * \brief   Reshaping Mapper for Matrix containers.
 * \author  Johann Briffa
 *
 * This class defines a symbol mapper where matrix reshaping occurs by reading
 * and writing elements in row-major order. Note that the input and output
 * alphabet sizes must be the same.
 */

template <class dbl>
class map_reshape : public mapper<libbase::matrix, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::matrix, dbl> Base;
   typedef map_reshape<dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::matrix<array1d_t> array2vd_t;
   // @}

private:
   /*! \name Internal object representation */
   libbase::size_type<libbase::matrix> size_out;
   // @}

protected:
   // Interface with mapper
   /*! \copydoc mapper::setup()
    *
    * \note Symbol alphabets must be the same size
    */
   void setup()
      {
      assertalways(Base::M == Base::q);
      }
   void dotransform(const array2i_t& in, array2i_t& out) const;
   void dotransform(const array2vd_t& pin, array2vd_t& pout) const;
   void doinverse(const array2vd_t& pin, array2vd_t& pout) const;

public:
   // Informative functions
   libbase::size_type<libbase::matrix> output_block_size() const
      {
      return size_out;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_reshape)
};

} // end namespace

#endif
