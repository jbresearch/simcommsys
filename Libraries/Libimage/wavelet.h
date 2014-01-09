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

#ifndef __wavelet_h
#define __wavelet_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include <iostream>

/*
 Version 1.00 (6 Jun 2000)
 Initial version - uses Daub4 wavelets and does single and 2-dimensional
 transforms and inverses.

 Version 1.10 (11 Nov 2001)
 Added number of levels to be performed for transform & inverse, and a public function
 that allows the user to find the size of the low-freq coefficients with that level

 Version 1.11 (6 Mar 2002)
 changed vcs version variable from a global to a static class variable.
 also changed use of iostream from global to std namespace.
 edited the classes to be compileable with Microsoft extensions enabled - in practice,
 the major change is in for() loops, where MS defines scope differently from ANSI.
 Here we chose to take the loop variables into function scope.

 Version 1.12 (13 Apr 2002)
 modified transform and inverse to make use of new row/col insert/extract functions
 in matrix.

 Version 1.13 (14 Apr 2002)
 reorganized implementation file; made destructor virtual inline; made quadrature a
 static function; changed error-displays to assert statements.

 Version 1.20 (15 Apr 2002)
 changed the external and internal transform/inverse functions to allow the user to
 specify both an input and an output vector/matrix. This allows us easily to keep
 the input data if we want; otherwise, the user can always specify both to be the
 same (the algorithm allows this). Also commented out the translation-invariant
 partial transform function since we're not using that at the moment anyway.

 Version 1.30 (30 Apr 2002)
 added support for Haar, Beylkin, Coiflet, Daubechies, Symmlet, Vaidyanathan, and
 Battle-Lemarie wavelets with a number of parameters for each. Also added support
 for default construction and later initialization.

 Version 1.40 (10 Nov 2006)
 * defined class and associated data within "libimage" namespace.
 * removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

namespace libimage {

class wavelet {
protected:
   // the quadrature mirror filters
   libbase::vector<double> g, h;
protected:
   // from the [smoothing] filter 'g' generate the quadrature [detail] filter 'h'
   static libbase::vector<double> quadrature(const libbase::vector<double>& g);
   // partial forward and inverse transforms
   void partial_transform(const libbase::vector<double>& in, libbase::vector<
         double>& out, const int n) const;
   void partial_inverse(const libbase::vector<double>& in, libbase::vector<
         double>& out, const int n) const;
   // partial translation-invariant transforms
   //void partial_titransform(vector<double>& a, vector<double>& hsr, vector<double>& hsl, vector<double>& lsr, vector<double>& lsl) const;
public:
   wavelet()
      {
      }
   wavelet(const int type, const int par = 0)
      {
      init(type, par);
      }
   virtual ~wavelet()
      {
      }

   void init(const int type, const int par = 0);

   int getlimit(const int size, const int level) const;

   void transform(const libbase::vector<double>& in,
         libbase::vector<double>& out, const int level = 0) const;
   void inverse(const libbase::vector<double>& in,
         libbase::vector<double>& out, const int level = 0) const;

   void transform(const libbase::matrix<double>& in,
         libbase::matrix<double>& out, const int level = 0) const;
   void inverse(const libbase::matrix<double>& in,
         libbase::matrix<double>& out, const int level = 0) const;
};

} // end namespace

#endif
