#include "logrealfast.h"

namespace libbase {

const vcs logrealfast::version("Fast Logarithm Arithmetic module (logrealfast)", 1.31);

const int logrealfast::lutsize = 1<<17;
const double logrealfast::lutrange = 12.0;
double *logrealfast::lut;
bool logrealfast::lutready = false;
#ifdef DEBUG
std::ofstream logrealfast::file;
#endif

// LUT constructor

void logrealfast::buildlut()
   {
   lut = new double[lutsize];
   for(int i=0; i<lutsize; i++)
      lut[i] = log(1 + exp(-lutrange*i/(lutsize-1)));
   lutready = true;
#ifdef DEBUG
   file.open("logrealfast-table.txt");
   file.precision(6);
#endif
   }

// conversion

double logrealfast::convertfromdouble(const double m)
   {
   using std::clog;
   // trap infinity
   const int inf = isinf(m);
   if(inf < 0)
      {
#ifdef DEBUG
      clog << "WARNING (logrealfast): -Infinity values cannot be represented (" << m << "); assuming infinitesimally small value.\n";
#endif
      return DBL_MAX;
      }
   else if(inf > 0)
      {
#ifdef DEBUG
      clog << "WARNING (logrealfast): +Infinity values cannot be represented (" << m << "); assuming infinitesimally large value.\n";
#endif
      return -DBL_MAX;
      }
   // trap NaN
   else if(isnan(m))
      {
#ifdef DEBUG
      clog << "WARNING (logrealfast): NaN values cannot be represented (" << m << "); assuming infinitesimally small value.\n";
#endif
      return DBL_MAX;
      }
   // trap negative numbers & zero
   else if(m <= 0)
      {
#ifdef DEBUG
      clog << "WARNING (logrealfast): Negative numbers and zero cannot be represented (" << m << "); assuming infinitesimally small value.\n";
#endif
      return DBL_MAX;
      }
   // finally convert (value must be ok)
   else 
      return -log(m);
   }

// Input/Output Operations

std::ostream& operator<<(std::ostream& s, const logrealfast& x)
   {        
   using std::ios;

   const double lg = -x.logval/log(10.0);

   const ios::fmtflags flags = s.flags();
   s.setf(ios::fixed, ios::floatfield);
   s << ::pow(10.0, lg-floor(lg));
   s.setf(ios::showpos);
   s << "e" << int(floor(lg));
   s.flags(flags);

   return s;
   }

}; // end namespace
