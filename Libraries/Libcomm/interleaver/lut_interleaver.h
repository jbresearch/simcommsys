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

#ifndef __lut_interleaver_h
#define __lut_interleaver_h

#include "config.h"
#include "interleaver.h"
#include "serializer.h"
#include "fsm.h"

namespace libcomm {

/*!
 * \brief   Lookup Table Interleaver.
 * \author  Johann Briffa
 *
 * \todo Document concept of forced tail interleavers (as in divs95)
 */

template <class real>
class lut_interleaver : public interleaver<real> {
protected:
   lut_interleaver()
      {
      }
   libbase::vector<int> lut;
public:
   virtual ~lut_interleaver()
      {
      }

   // Transform functions
   void
   transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void
   transform(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;
   void
   inverse(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;

   // Information functions
   int size() const
      {
      return lut.size();
      }
};

} // end namespace

#endif

