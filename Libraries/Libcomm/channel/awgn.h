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

#ifndef __awgn_h
#define __awgn_h

#include "config.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   Additive White Gaussian Noise Channel.
 * \author  Johann Briffa
 *
 * \version 1.10 (15 Apr 1999)
 * Changed the definition of set_snr to avoid using the pow() function.
 * This was causing an unexplained SEGV with optimised code
 *
 * \version 1.11 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 *
 * \version 1.20 (13 Mar 2002)
 * updated the system to conform with the completed serialization protocol (in conformance
 * with channel 1.10), by adding the necessary name() function, and also by adding a static
 * serializer member and initialize it with this class's name and the static constructor
 * (adding that too). Also made the channel object a public base class, rather than a
 * virtual public one, since this was affecting the transfer of virtual functions within
 * the class (causing access violations). Also moved most functions into the implementation
 * file rather than here.
 *
 * \version 1.30 (27 Mar 2002)
 * changed descriptive output function to conform with channel 1.30.
 *
 * \version 1.40 (30 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 *
 * \version 1.50 (16 Oct 2007)
 * changed class to conform with channel 1.50.
 *
 * \version 1.51 (16 Oct 2007)
 * changed class to conform with channel 1.51.
 *
 * \version 1.52 (17 Oct 2007)
 * changed class to conform with channel 1.52.
 *
 * \version 1.53 (29 Oct 2007)
 * - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
 *
 * \version 1.54 (24 Jan 2008)
 * - Changed derivation from channel to channel<sigspace>
 */

class awgn : public channel<sigspace> {
   // channel paremeters
   double sigma;
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(awgn)
};

} // end namespace

#endif

