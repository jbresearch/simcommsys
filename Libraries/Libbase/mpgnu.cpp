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

#include "mpgnu.h"
#include <cstdlib>

namespace libbase {

#ifdef USE_GMP
mpf_t mpgnu::dblmin;
mpf_t mpgnu::dblmax;
#endif

void mpgnu::init()
   {
#ifndef USE_GMP
   std::cerr
         << "FATAL ERROR (mpgnu): GNU Multi-Precision not implemented - cannot initialise." << std::endl;
   exit(1);
#else
   static bool ready = false;
   if(!ready)
      {
      mpf_init_set_d(dblmin, DBL_MIN);
      mpf_init_set_d(dblmax, DBL_MAX);
      ready = true;
      }
#endif
   }

} // end namespace
