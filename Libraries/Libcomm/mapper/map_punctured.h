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

#ifndef __map_punctured_h
#define __map_punctured_h

#include "mapper.h"

namespace libcomm {

/*!
 * \brief   Punctured Mapper - Template base.
 * \author  Johann Briffa
 *
 * This class is a template definition for punctured mappers; this needs to
 * be specialized for actual use. Template parameter defaults are provided
 * here.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_punctured : public mapper<C, dbl> {
};

/*!
 * \brief   Punctured Mapper - Vector containers.
 * \author  Johann Briffa
 *
 * This class defines a general punctured mapper where the puncturing matrix
 * is directly specified by the user. This puncturing matrix is repeatedly
 * applied to the encoded output, the length of which must be an exact
 * multiple of the puncturing matrix size.
 */

template <class dbl>
class map_punctured<libbase::vector, dbl> : public mapper<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector, dbl> Base;
   typedef map_punctured<libbase::vector, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

private:
   /*! \name User-defined parameters */
   libbase::matrix<bool> punc_matrix; //!< User-defined puncturing matrix
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
      // shorthand for puncturing matrix rate (p of P)
      const int p = libbase::matrix<int>(punc_matrix).sum();
      const int P = punc_matrix.size();
      // compute rate
      return p / double(P);
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      // shorthand for puncturing matrix rate (p of P)
      const int p = libbase::matrix<int>(punc_matrix).sum();
      const int P = punc_matrix.size();
      // find out how many times the puncturing matrix fits
      const int n = size.length() / P;
      assert(size.length() == n * P);
      // compute output block size
      return libbase::size_type<libbase::vector>(n * p);
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_punctured)
};

} // end namespace

#endif
