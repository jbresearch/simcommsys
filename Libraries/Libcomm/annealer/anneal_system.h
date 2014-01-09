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

#ifndef __anneal_system_h
#define __anneal_system_h

#include "config.h"
#include "random.h"
#include <iostream>

namespace libcomm {

/*!
 * \brief   Simulated Annealing System base.
 * \author  Johann Briffa
 *
 * \version 1.00 (10 Jul 1998)
 *
 * \version 1.10 (11 Oct 2001)
 * added a virtual function which outputs the annealed system
 * (this was only done before in the destruction mechanism)
 *
 * \version 1.11 (26 Oct 2001)
 * added a virtual destroy function (see interleaver.h)
 *
 * \version 1.20 (4 Nov 2001)
 * added a stream << operator and modified the regular output routine accordingly
 * (so that this now returns the stream, and is a pure virtual).
 *
 * \version 1.21 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.22 (17 Jul 2006)
 * added virtual destructor, since this is not done by default.
 *
 * \version 1.23 (25 Jul 2006)
 * added empty definition for virtual destructor.
 *
 * \version 1.30 (27 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class anneal_system {
public:
   virtual ~anneal_system()
      {
      }
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r) = 0;
   //! Perturbs the state and returns the difference in energy due to perturbation
   virtual double perturb() = 0;
   //! Undoes the last perturbation (guaranteed only for one stage)
   virtual void unperturb() = 0;
   //! Returns the system's energy content
   virtual double energy() = 0;
   //! Outputs the system to an output stream
   virtual std::ostream& output(std::ostream& sout) const = 0;
   friend std::ostream& operator<<(std::ostream& sout, const anneal_system& x);
};

} // end namespace

#endif
