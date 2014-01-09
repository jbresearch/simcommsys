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

#ifndef __waveletfilter_h
#define __waveletfilter_h

#include "filter.h"
#include "wavelet.h"
#include <list>
#include <vector>

/*
 Version 1.00 (11 Nov 2001)
 Initial version.

 Version 1.01 (6 Mar 2002)
 changed vcs version variable from a global to a static class variable.

 Version 1.02 (14 Mar 2002)
 reorganized implementation file; made destructor inline.

 Version 1.10 (15 Apr 2002)
 class is now derived from filter. Also fixed a couple bugs in the thresholding
 process while doing the conversion:
 * when finding the threshold from the histogram I was (probably - the histogram classes
 are a little messy & need to be updated/changed/removed) only considering the positive
 coefficients; now I work on the absolute values.
 * for a transform where the lower (xlimit,ylimit) coefficients are the low-frequency
 ones, only the coefficients at x>xlimit, y>ylimit were being shrunk. This is not the
 same as "all high-frequency coeffs", and was introducing some artifacts. It is now
 fixed.

 Version 1.11 (21 Apr 2002)
 added soft thresholding.

 Version 1.20 (29 Apr 2002)
 renamed internal variables; added visu threshold selection & paved the way for
 other selectors; added progress display routine in conformance with filter 1.10.
 also fixed a bug in % cutoff method - was computing %age on sorted list of coefficients
 and not on their absolute values, as it should be.

 Version 1.30 (30 Apr 2002)
 added support for Haar, Beylkin, Coiflet, Daubechies, Symmlet, Vaidyanathan, and
 Battle-Lemarie wavelets with a number of parameters for each, as in wavelet 1.30.

 Version 1.31 (13 Oct 2006)
 * added explicit conversion to int in round's output in estimate().
 * added explicit parameter conversion to double for log in estimate().

 Version 1.40 (10 Nov 2006)
 * defined class and associated data within "libimage" namespace.
 * removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

namespace libimage {

class waveletfilter : public filter<double> {
private:
   // user-supplied settings
   wavelet m_wWavelet;
   int m_nWaveletLevel;
   int m_nThreshSelector;
   int m_nThreshType;
   double m_dThreshCutoff;
   // internal variables
   std::vector<double> m_vdCoefficient;
   int m_nSize;
   double m_dThreshValue;
protected:
   void createmask(libbase::matrix<bool>& mask, const int xsize,
         const int ysize) const;
public:
   waveletfilter()
      {
      }
   waveletfilter(const int nType, const int nPar, const int nLevel,
         const int nThreshType, const int nThreshSelector,
         const double dThreshCutoff = 0)
      {
      init(nType, nPar, nLevel, nThreshType, nThreshSelector, dThreshCutoff);
      }
   virtual ~waveletfilter()
      {
      }
   // initialization
   void init(const int nType, const int nPar, const int nLevel,
         const int nThreshType, const int nThreshSelector,
         const double dThreshCutoff = 0);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset();
   void update(const libbase::matrix<double>& in);
   void estimate();
   // filter process loop (only updates output matrix)
   void
         process(const libbase::matrix<double>& in,
               libbase::matrix<double>& out) const;
};

} // end namespace

#endif
