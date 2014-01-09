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

#ifndef __anneal_interleaver_h
#define __anneal_interleaver_h

#include "config.h"
#include "matrix.h"
#include "randgen.h"
#include "annealer/anneal_system.h"

namespace libcomm {

/*!
 * \brief   Simulated Annealing Interleaver Design.
 * \author  Johann Briffa
 *
 * \version 1.00
 * Six different energy functions were used, as detailed in my Feb 1 report "On the importance
 * of the interleaver in parallel concatenated turbo codes".
 *
 * \version 1.10
 * Removed the bug mentioned in the report (affecting the range of j) for type 5 function [type 7]
 *
 * \version 1.20
 * Removed the restriction on the positions to swap for correct tailing (ie. use as untailed) [type 8]
 * also speeded up annealing by introducing the delta function (optimised)
 *
 * \version 1.30 (27 Apr 1999)
 * New energy function [type 9]: SUM of 5*m/r
 *
 * \version 1.40 (28 Apr 1999)
 * New energy function [type 10]: SUM of ( (tau*sqrt(2) - r)/(tau*sqrt(2)) ) ^ 4
 *
 * \version 1.50 (30 Apr 1999)
 * Changed energy function [type 11]: basically same as type 9, but corrected for annulus area
 *
 * \version 1.60 (3 May 1999)
 * New energy function [type 12]: still using corrected weighting, but now we have a function
 * that is symmetric across the valid range of r
 *
 * \version 1.70 (6 May 1999)
 * New energy function [type 13]: same as type 9, but corrected for 'flat' distribution
 *
 * \version 2.00 (25 May 1999)
 * Integrated different energy functions into a single class, user selected at creation.
 * Also, on creation, the LUT is randomised.
 *
 * \version 2.10 (28 Sep 1999)
 * Added support for self-terminating interleavers.
 *
 * \version 2.20 (2 Oct 1999)
 * New energy function [type 14]: SUM of log(l_in + l_out) where the two lengths are the lengths
 * of the zero sequences in the parity for input and interleaved sequences, based on the impulse
 * response of the encoder.
 *
 * \version 3.00 (12 Mar 2000)
 * Extended the annealer to support multi-dimensional interleavers, while keeping compatibility
 * and support for all old 2D interleaver types. New energy function [type 15] implements the
 * same thing as type 9 but scaled to multi-dimensions.
 *
 * \version 3.01 (15 Mar 2001)
 * Allowed the setting of the random seed during creation.
 *
 * \version 3.10 (11 Oct 2001)
 * added a virtual function which outputs the annealed system (this was only done
 * before in the destruction mechanism), in accordance with anneal_system 1.10
 *
 * \version 3.20 (2 Nov 2001)
 * Modified energy function - rather than working delta and then returning the value when
 * the end of the function is reached, we now return the delta energy immediately. Should
 * be slightly more efficient, and makes the creation of more complex energy functions
 * easier. Also created new energy function [type 16].
 *
 * \version 3.21 (4 Nov 2001)
 * modified the stream output function in accordance with anneal_system 1.20
 *
 * \version 3.22 (26 Feb 2002)
 * removed the automatic printing feature at destruction.
 *
 * \version 3.23 (1 Mar 2002)
 * edited the classes to be compileable with Microsoft extensions enabled - in practice,
 * the major change is in for() loops, where MS defines scope differently from ANSI.
 * Rather than taking the loop variables into function scope, we chose to avoid having
 * more than one loop per function, by defining private helper functions.
 *
 * \version 3.24 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 3.25 (11-12 Jun 2002)
 * - added "algorithm" to supply the definition of the swap() function (which has
 * been removed from config).
 * - moved the (empty) destructor to the definition file.
 *
 * \version 3.26 (6 Oct 2006)
 * modified for compatibility with VS .NET 2005:
 * - in energy_function, modified use of pow to avoid ambiguity
 *
 * \version 3.30 (27 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 *
 * \version 3.31 (2 Jan 2008)
 * - modified stream output to include only LUT contents, not index
 */

class anneal_interleaver : public virtual anneal_system {
   libbase::matrix<int> lut;
   libbase::randgen r;
   bool term;
   int sets, tau, m, type;
   double f0, f1, f2;
   double E, Eold;
   int set, pos1, pos2;
protected:
   void initialise();
   double energy_function(const int i, const int j);
   double work_energy();
   double work_delta();
public:
   anneal_interleaver(const int sets, const int tau, const int m,
         const int type, const bool term);
   ~anneal_interleaver()
      {
      }
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
