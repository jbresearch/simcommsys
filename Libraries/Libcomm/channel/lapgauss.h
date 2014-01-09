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

#ifndef __lapgauss_h
#define __lapgauss_h

#include "config.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   Additive Laplacian-Gaussian Channel.
 * \author  Johann Briffa
 *
 * \todo this class is still unfinished, and only implements the plain
 * Gaussian channel right now
 */

class lapgauss : public channel<sigspace> {
   // channel paremeters
   double sigma;
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // object handling
   lapgauss();

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(lapgauss)
};

} // end namespace

#endif

