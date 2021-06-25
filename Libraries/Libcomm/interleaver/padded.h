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

#ifndef __padded_h
#define __padded_h

#include "config.h"
#include "interleaver.h"
#include "onetimepad.h"
#include "serializer.h"
#include <iostream>

namespace libcomm {

/*!
 * \brief   Padded Interleaver.
 * \author  Johann Briffa
 *
 * \note The member onetimepad object is a pointer; this allows us to create
 * an empty "padded" class without access to onetimepad's default
 * constructor (which is private for that class).
 */

template <class real>
class padded : public interleaver<real> {
   std::shared_ptr<interleaver<real> > otp;
   std::shared_ptr<interleaver<real> > inter;
protected:
   padded()
      {
      }
public:
   padded(const interleaver<real>& inter, const fsm& encoder,
         const bool terminated, const bool renewable);
   padded(const padded& x);
   ~padded()
      {
      }

   // Intra-frame Operations
   void seedfrom(libbase::random& r);
   void advance();

   // Transform functions
   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void transform(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;
   void inverse(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;

   // Information functions
   int size() const
      {
      assertalways(inter);
      return inter->size();
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(padded)
};

} // end namespace

#endif
