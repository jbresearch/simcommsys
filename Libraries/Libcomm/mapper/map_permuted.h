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

#ifndef __map_permuted_h
#define __map_permuted_h

#include "mapper.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
 * \brief   Random Symbol Permutation Mapper - Template base.
 * \author  Johann Briffa
 *
 * This class is a template definition for random symbol-permuting mappers;
 * this needs to be specialized for actual use. Template parameter defaults are
 * provided here.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_permuted : public mapper<C, dbl> {
};

/*!
 * \brief   Random Symbol Permutation Mapper - Vector containers.
 * \author  Johann Briffa
 *
 * This class defines a symbol-permuting mapper.
 */

template <class dbl>
class map_permuted<libbase::vector, dbl> : public mapper<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector, dbl> Base;
   typedef map_permuted<libbase::vector, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

private:
   /*! \name Internal object representation */
   mutable libbase::vector<libbase::randperm> lut;
   mutable libbase::randgen r;
   // @}

protected:
   // Pull in base class variables
   using Base::M;
   using Base::q;

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const array1i_t& in, array1i_t& out) const;
   void dotransform(const array1vd_t& pin, array1vd_t& pout) const;
   void doinverse(const array1vd_t& pin, array1vd_t& pout) const;

public:
   // Setup functions
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_permuted)
};

} // end namespace

#endif
