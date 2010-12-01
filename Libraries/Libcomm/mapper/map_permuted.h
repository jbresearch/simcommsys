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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __map_permuted_h
#define __map_permuted_h

#include "map_straight.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
 * \brief   Random Symbol Permutation Mapper.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class defines an symbol-permuting version of the straight mapper.
 *
 * \todo Make this class inherit from any base mapper, not just straight
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_permuted : public map_straight<C, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef map_straight<C, dbl> Base;
   typedef map_permuted<C, dbl> This;

private:
   /*! \name Internal object representation */
   mutable libbase::vector<libbase::randperm> lut;
   mutable libbase::randgen r;
   // @}

protected:
   // Pull in base class variables
   using Base::M;

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const C<int>& in, C<int>& out) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

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
