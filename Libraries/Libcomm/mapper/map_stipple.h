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

#ifndef __map_stipple_h
#define __map_stipple_h

#include "mapper.h"

namespace libcomm {

/*!
 * \brief   Stipple Mapper - Template base.
 * \author  Johann Briffa
 *
 * This class is a template definition for stipple mappers; this needs to
 * be specialized for actual use. Template parameter defaults are provided
 * here.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_stipple : public mapper<C, dbl> {
};

/*!
 * \brief   Stipple Mapper - Vector containers.
 * \author  Johann Briffa
 *
 * This class defines a punctured mapper suitable for turbo codes, where:
 * - all information symbols are transmitted
 * - parity symbols are taken from successive sets
 * This results in an overall rate of 1/2
 * For a two-set turbo code, this corresponds to odd/even puncturing.
 */

template <class dbl>
class map_stipple<libbase::vector, dbl> : public mapper<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector, dbl> Base;
   typedef map_stipple<libbase::vector, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

private:
   /*! \name User-defined parameters */
   int sets; //!< Number of turbo code parallel sets
   // @}
   /*! \name Internal object representation */
   mutable libbase::vector<bool> pattern; //!< Pre-computed puncturing pattern
   // @}

protected:
   // Pull in base class variables
   using Base::size;
   using Base::q;
   using Base::M;

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const array1i_t& in, array1i_t& out) const;
   void dotransform(const array1vd_t& pin, array1vd_t& pout) const;
   void doinverse(const array1vd_t& pin, array1vd_t& pout) const;

public:
   // Informative functions
   double rate() const
      {
      return (sets + 1) / 2.0;
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      const int tau = size.length() / (sets + 1);
      assert(size.length() == tau * (sets + 1));
      return libbase::size_type<libbase::vector>(tau * 2);
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_stipple)
};

} // end namespace

#endif
