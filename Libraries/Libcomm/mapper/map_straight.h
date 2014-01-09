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

#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
 * \brief   Straight Mapper - Template base.
 * \author  Johann Briffa
 *
 * This class is a template definition for straight mappers; this needs to
 * be specialized for actual use. Template parameter defaults are provided
 * here.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_straight : public mapper<C, dbl> {
};

/*!
 * \brief   Straight Mapper - Vector containers.
 * \author  Johann Briffa
 *
 * This class defines a straight symbol mapper; this is a rate-1 mapper
 * for cases where each modulation symbol encodes exactly one encoder symbol.
 */

template <class dbl>
class map_straight<libbase::vector, dbl> : public mapper<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector, dbl> Base;
   typedef map_straight<libbase::vector, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;
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
   void dotransform(const array1i_t& in, array1i_t& out) const
      {
      out = in;
      }
   void dotransform(const array1vd_t& pin, array1vd_t& pout) const
      {
      pout = pin;
      }
   void doinverse(const array1vd_t& pin, array1vd_t& pout) const
      {
      pout = pin;
      }

public:
   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER (map_straight)
};

/*!
 * \brief   Straight Mapper - Matrix containers.
 * \author  Johann Briffa
 *
 * This class defines a straight symbol mapper; this is a rate-1 mapper
 * for cases where each modulation symbol encodes exactly one encoder symbol.
 * Additionally, for matrix containers, the encoder output and blockmodem
 * input containers have the same shape.
 */

template <class dbl>
class map_straight<libbase::matrix, dbl> : public mapper<libbase::matrix, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::matrix, dbl> Base;
   typedef map_straight<libbase::matrix, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::matrix<array1d_t> array2vd_t;
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
   void dotransform(const array2i_t& in, array2i_t& out) const
      {
      out = in;
      }
   void dotransform(const array2vd_t& pin, array2vd_t& pout) const
      {
      pout = pin;
      }
   void doinverse(const array2vd_t& pin, array2vd_t& pout) const
      {
      pout = pin;
      }

public:
   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER (map_straight)
};

} // end namespace

#endif
