/*!
 * \file
 * $Id: qam.h 9469 2013-07-24 16:38:03Z jabriffa $
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

#ifndef __conv_h
#define __conv_h

#include "lut_modulator.h"
#include "config.h"
#include "stream_modulator.h"
#include "channel/qids.h"
#include "algorithm/fba2-interface.h"
#include "randgen.h"
#include "itfunc.h"
#include "vector_itfunc.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>
#include <memory>

#include "boost/shared_ptr.hpp"

namespace libcomm {

/*!
 * \brief   QAM Modulator.
 * \author  Johann Briffa
 * $Id: qam.h 9469 2013-07-24 16:38:03Z jabriffa $
 *
 * \version 1.00 (3 Jan 2008)
 * - Initial version, implements square QAM with Gray-coded mapping
 * - Derived from mpsk 2.20
 */

class conv : public lut_modulator {
public: 
   /*Type Definitions*/
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<int> array1s_t;

protected:
   /*! \name Internal operations */
   void init(const int m);
   void domodulate(const int N, const array1i_t& encoded, array1s_t& tx);

public:
   conv()
      {
      }

   conv(const int m)
      {
      init(m);
      }
   ~conv()
      {
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(conv)
};

} // end namespace

#endif
