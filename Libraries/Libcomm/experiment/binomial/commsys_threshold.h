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

#ifndef __commsys_threshold_h
#define __commsys_threshold_h

#include "config.h"
#include "commsys_simulator.h"

namespace libcomm {

/*!
 * \brief   Communication System Simulator - Variation of modem threshold.
 * \author  Johann Briffa
 *
 * A variation on the regular commsys_simulator object, taking a fixed channel
 * parameter and varying modem threshold.
 *
 * \todo Remove assumption of a dminner-derived modem.
 */
template <class S, class R>
class commsys_threshold : public commsys_simulator<S, R> {
private:
   // Shorthand for class hierarchy
   typedef commsys_threshold<S, R> This;
   typedef commsys_simulator<S, R> Base;

public:
   // Experiment parameter handling
   void set_parameter(const double x);
   double get_parameter() const;

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(commsys_threshold)
};

} // end namespace

#endif
