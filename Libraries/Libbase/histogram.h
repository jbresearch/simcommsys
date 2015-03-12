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

#ifndef __histogram_h
#define __histogram_h

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
 * \brief   Histogram.
 * \author  Johann Briffa
 *
 * Computes the histogram of the values in a vector or matrix with the
 * user-supplied number of bins.
 */

class histogram {
   double step;
   vector<double> x;
   vector<int> y;
private:
   void initbins(const double min, const double max, const int n);
public:
   histogram(const vector<double>& a, const int n);

   int bins() const
      {
      return x.size();
      }
   int freq(const int i) const
      {
      return y(i);
      }
   double val(const int i) const
      {
      return x(i) + step / 2;
      }

   double max() const
      {
      return x(x.size() - 1) + step;
      }
   double min() const
      {
      return x(0);
      }
};

class phistogram {
   double step;
   vector<double> x, y;
private:
   static double findmax(const matrix<double>& a);
   void initbins(const double max, const int n);
   void accumulate();
public:
   phistogram(const matrix<double>& a, const int n);

   int bins() const
      {
      return x.size();
      }
   double freq(const int i) const
      {
      return y(i);
      }
   double val(const int i) const
      {
      return x(i);
      }
};

class chistogram {
   double step;
   vector<double> x, y;
private:
   static double findmax(const matrix<double>& a);
   static double findmax(const matrix<double>& a, const matrix<bool>& mask);
   static int count(const matrix<bool>& mask);
   void initbins(const double max, const int n);
   void accumulate();
public:
   chistogram(const matrix<double>& a, const int n);
   chistogram(const matrix<double>& a, const matrix<bool>& mask, const int n);

   int bins() const
      {
      return x.size();
      }
   double freq(const int i) const
      {
      return y(i);
      }
   double val(const int i) const
      {
      return x(i);
      }
};

} // end namespace

#endif

