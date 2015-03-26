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

#ifndef __map_dividing_h
#define __map_dividing_h

#include "mapper.h"
#include "symbol_converter.h"

namespace libcomm {

/*!
 * \brief   Dividing Mapper - Template base.
 * \author  Johann Briffa
 *
 * This class is a template definition for dividing mappers; this needs to
 * be specialized for actual use. Template parameter defaults are provided
 * here.
 *
 * \tparam dbl2 Floating-point type for internal computation (pre-normalization)
 */

template <template <class > class C = libbase::vector, class dbl = double, class dbl2 = double>
class map_dividing : public mapper<C, dbl> {
};

/*!
 * \brief   Dividing Mapper - Vector containers.
 * \author  Johann Briffa
 *
 * This class defines a dividing symbol mapper; this is a rate-1 mapper for
 * cases where each encoder symbol is represented by more than one modulation
 * symbol. For example, it will allow the use of q-ary codecs on binary
 * modulators.
 *
 * \note Each encoder output symbol must be representable by an integral
 * number of modulation symbols
 */

template <class dbl, class dbl2>
class map_dividing<libbase::vector, dbl, dbl2> : public mapper<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector, dbl> Base;
   typedef map_dividing<libbase::vector, dbl, dbl2> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

protected:
   // Interface with mapper
   void dotransform(const array1i_t& in, array1i_t& out) const;
   void dotransform(const array1vd_t& pin, array1vd_t& pout) const;
   void doinverse(const array1vd_t& pin, array1vd_t& pout) const;

public:
   // Informative functions
   libbase::size_type<libbase::vector> output_block_size() const
      {
      const int k = libbase::symbol_converter<dbl, dbl2>::get_rate(Base::M,
            Base::q);
      return libbase::size_type<libbase::vector>(this->size * k);
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_dividing)
};

} // end namespace

#endif
