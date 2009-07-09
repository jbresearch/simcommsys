/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "histogram.h"
#include <float.h>

namespace libbase {

void histogram::initbins(const double min, const double max, const int n)
   {
   step = (max - min) / double(n);
   x.init(n);
   for (int i = 0; i < n; i++)
      x(i) = min + i * step;
   }

histogram::histogram(const vector<double>& a, const int n)
   {
   initbins(a.min(), a.max(), n);

   y.init(n);
   y = 0;
   double sum = 0, sumsq = 0;
   for (int i = 0; i < a.size(); i++)
      {
      sum += a(i);
      sumsq += a(i) * a(i);
      for (int k = n - 1; k >= 0; k--)
         if (a(i) >= x(k))
            {
            y(k)++;
            break;
            }
      }
   int N = a.size();
   mean = sum / N;
   var = sumsq / N - mean * mean;
   }

histogram::histogram(const matrix<double>& a, const int n)
   {
   initbins(a.min(), a.max(), n);

   y.init(n);
   y = 0;
   double sum = 0, sumsq = 0;
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         {
         sum += a(i, j);
         sumsq += a(i, j) * a(i, j);
         for (int k = n - 1; k >= 0; k--)
            if (a(i, j) >= x(k))
               {
               y(k)++;
               break;
               }
         }
   int N = a.size();
   mean = sum / N;
   var = sumsq / N - mean * mean;
   }

double phistogram::findmax(const matrix<double>& a)
   {
   double max = a(0, 0);
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         if (fabs(a(i, j)) > max)
            max = fabs(a(i, j));
   return max;
   }

void phistogram::initbins(const double max, const int n)
   {
   step = max / double(n);

   x.init(n);
   for (int i = 0; i < n; i++)
      x(i) = (i + 1) * step;
   }

void phistogram::accumulate()
   {
   for (int i = 1; i < y.size(); i++)
      y(i) += y(i - 1);
   }

phistogram::phistogram(const matrix<double>& a, const int n)
   {
   initbins(findmax(a), n);

   y.init(n);
   y = 0;
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         for (int k = 0; k < n; k++)
            if (fabs(a(i, j)) <= x(k))
               {
               y(k) += a(i, j) * a(i, j);
               break;
               }

   // normalise & make cumulative
   y /= a.sumsq();
   accumulate();
   }

double chistogram::findmax(const matrix<double>& a)
   {
   double max = fabs(a(0, 0));
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         if (fabs(a(i, j)) > max)
            max = fabs(a(i, j));
   return max;
   }

double chistogram::findmax(const matrix<double>& a, const matrix<bool>& mask)
   {
   double max = -DBL_MAX;
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         if (mask(i, j))
            {
            if (fabs(a(i, j)) > max)
               max = fabs(a(i, j));
            }
   return max;
   }

int chistogram::count(const matrix<bool>& mask)
   {
   int elements = 0;
   for (int i = 0; i < mask.size().rows(); i++)
      for (int j = 0; j < mask.size().cols(); j++)
         if (mask(i, j))
            elements++;
   return elements;
   }

void chistogram::initbins(const double max, const int n)
   {
   step = max / double(n);

   x.init(n);
   for (int i = 0; i < n; i++)
      x(i) = (i + 1) * step;
   }

void chistogram::accumulate()
   {
   for (int i = 1; i < y.size(); i++)
      y(i) += y(i - 1);
   }

chistogram::chistogram(const matrix<double>& a, const int n)
   {
   initbins(findmax(a), n);

   double delta = 1 / double(a.size());

   y.init(n);
   y = 0;
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         for (int k = 0; k < n; k++)
            if (fabs(a(i, j)) <= x(k))
               {
               y(k) += delta;
               break;
               }

   accumulate();
   }

chistogram::chistogram(const matrix<double>& a, const matrix<bool>& mask,
      const int n)
   {
   initbins(findmax(a, mask), n);

   const double delta = 1 / double(count(mask));
   y.init(n);
   y = 0;
   for (int i = 0; i < a.size().rows(); i++)
      for (int j = 0; j < a.size().cols(); j++)
         if (mask(i, j))
            for (int k = 0; k < n; k++)
               if (fabs(a(i, j)) <= x(k))
                  {
                  y(k) += delta;
                  break;
                  }

   accumulate();
   }

} // end namespace
