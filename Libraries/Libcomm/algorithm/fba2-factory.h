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

#ifndef __fba2_factory_h
#define __fba2_factory_h

#include "fba2-interface.h"

#include "boost/shared_ptr.hpp"

namespace libcomm {

/*!
 * \brief   Factory to return the desired Symbol-Level Forward-Backward
 *          Algorithm (for TVB codes).
 * \author  Johann Briffa
 *
 * This factory automatically chooses between CPU and GPU implementations
 * depending on compiler flags, and takes as parameters the flag values
 * (which are determined at runtime).
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 */

template <class sig, class real, class real2>
class fba2_factory {
public:
   //! Return an instance of the FBA2 algorithm
   static boost::shared_ptr<fba2_interface<sig, real, real2> > get_instance(
         bool fss, bool thresholding, bool lazy, bool globalstore);
};

} // end namespace

#endif
