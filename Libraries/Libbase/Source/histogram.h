#ifndef __histogram_h
#define __histogram_h
      
#include "config.h"
#include "vcs.h"
#include "vector.h"
#include "matrix.h"
#include <math.h>
#include <iostream>

namespace libbase {

/*!
   \brief   Histogram.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00 (8 Jun 2000)
  Initial version - computes the histogram of the values in a vector or
  matrix with the user-supplied number of bins.

  Version 1.10 (31 Oct 2001)
  modified the internal code to make use of the new enhanced matrix and vector classes.

  Version 1.20 (11 Nov 2001)
  modified the internal code to conform with the min/max renaming in matrix & vector.
  Also added a new creation routine for chistogram that makes use of a mask matrix -
  this allows the user to mask out parts of a matrix from the histogram.

  Version 1.21 (1 Mar 2002)
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Rather than taking the loop variables into function scope, we chose to avoid having
  more than one loop per function, by defining private helper functions.

  Version 1.22 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.30 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class histogram {
   static const vcs version;
   double step, mean, var;
   vector<double> x;
   vector<int> y;
private:
   void initbins(const double min, const double max, const int n);
public:
   histogram(const vector<double>& a, const int n);
   histogram(const matrix<double>& a, const int n);

   int bins() const { return x.size(); }
   int freq(const int i) const { return y(i); }
   double val(const int i) const { return x(i) + step/2; }

   double mu() const { return mean; }
   double sigma() const { return sqrt(var); }

   double max() const { return x(x.size()-1)+step; }
   double min() const { return x(0); }
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

   int bins() const { return x.size(); }
   double freq(const int i) const { return y(i); }
   double val(const int i) const { return x(i); }
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

   int bins() const { return x.size(); }
   double freq(const int i) const { return y(i); }
   double val(const int i) const { return x(i); }
};

}; // end namespace

#endif

