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

#ifndef __sysrepacc_h
#define __sysrepacc_h

#include "config.h"
#include "repacc.h"

namespace libcomm {

/*!
 * \brief   Systematic Repeat-Accumulate (SRA) codes.
 * \author  Johann Briffa
 *
 * Extension of the Repeat-Accumulate (RA) codes, also transmitting
 * systematic data on the channel.
 */

template <class real, class dbl = double>
class sysrepacc : public repacc<real, dbl> {
private:
   // Shorthand for class hierarchy
   typedef sysrepacc<real, dbl> This;
   typedef repacc<real, dbl> Base;
   typedef safe_bcjr<real, dbl> BCJR;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<dbl> array2d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

private:
   // Grant access to inherited fields and methods
   using Base::ra;
   using Base::rp;
   using Base::R;
   using Base::initialised;
   using Base::allocate;
   using Base::reset;

protected:
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
   void do_init_decoder(const array1vd_t& ptable);
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app);

public:
   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> output_block_size() const
      {
      // Inherit sizes
      const int Ns = Base::input_block_size();
      const int Np = Base::output_block_size();
      return libbase::size_type<libbase::vector>(Ns + Np);
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(sysrepacc)
};

} // end namespace

#endif
