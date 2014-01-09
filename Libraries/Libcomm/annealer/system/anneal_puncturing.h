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

#ifndef __anneal_puncturing_h
#define __anneal_puncturing_h

#include "config.h"
#include "randgen.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"
#include "annealer/anneal_system.h"

namespace libcomm {

/*!
 * \brief   Simulated Annealing Puncturing Pattern Design.
 * \author  Johann Briffa
 *
 * \version 1.00 (5 Jun 1999)
 * In this version, we load the contribution of each data and parity bit from a file; then we try to
 * create a turbo code with the smallest variance of decoded-bit confidence. The restrictions are that
 * we will be puncturing half the bits of each parity sequence. We assume that initially we have a
 * rate-third code, and thus we will finish with a rate-half code. We also assume binary code types and
 * binary modulation for simplicity.
 *
 * \version 1.10 (11 Jun 1999)
 * Modified the algorithm by counting the variance of confidences produced by *punctured* bits, and
 * not by the unpunctured ones.
 *
 * \version 1.11 (4 Nov 2001)
 * added a stream output function in accordance with anneal_system 1.20
 *
 * \version 1.12 (1 Mar 2002)
 * edited the classes to be compileable with Microsoft extensions enabled - in practice,
 * the major change is in for() loops, where MS defines scope differently from ANSI.
 * Rather than taking the loop variables into function scope, we chose to wrap around the
 * offending for() loops.
 *
 * \version 1.13 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.20 (27 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class anneal_puncturing : public virtual anneal_system {
   libbase::matrix3<double> contrib;
   libbase::matrix<bool> pattern;
   libbase::vector<double> res;
   libbase::randgen r;
   int tau, s;
   double E, Eold;
   int set, pos1, pos2;
protected:
   void energy_function(const double factor, const int set, const int pos);
   double work_energy();
public:
   anneal_puncturing(const char *fname, const int tau, const int s);
   ~anneal_puncturing();
   // seeding for random generator
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }
   // perturb returns the difference in energy due to perturbation
   double perturb();
   void unperturb();
   double energy();
   // output the system
   std::ostream& output(std::ostream& sout) const;
};

} // end namespace

#endif
