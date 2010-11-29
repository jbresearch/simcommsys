/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "logrealfast.h"

namespace libbase {

const int logrealfast::lutsize = 1 << 17;
const double logrealfast::lutrange = 12.0;
double *logrealfast::lut;
bool logrealfast::lutready = false;
#ifdef DEBUGFILE
std::ofstream logrealfast::file;
#endif

// LUT constructor

void logrealfast::buildlut()
   {
   lut = new double[lutsize];
   for (int i = 0; i < lutsize; i++)
      lut[i] = log(1 + exp(-lutrange * i / (lutsize - 1)));
   lutready = true;
#ifdef DEBUGFILE
   file.open("logrealfast-table.txt");
   file.precision(6);
#endif
   }

// conversion

double logrealfast::convertfromdouble(const double m)
   {
   // trap infinity
   const int inf = isinf(m);
   if (inf < 0)
      {
#ifndef NDEBUG
      static int warningcount = 10;
      if (--warningcount > 0)
         trace
               << "WARNING (logrealfast): -Infinity values cannot be represented ("
               << m << "); assuming infinitesimally small value." << std::endl;
      else if (warningcount == 0)
         trace
               << "WARNING (logrealfast): last warning repeated too many times; stopped logging." << std::endl;
#endif
      return DBL_MAX;
      }
   else if (inf > 0)
      {
#ifndef NDEBUG
      static int warningcount = 10;
      if (--warningcount > 0)
         trace
               << "WARNING (logrealfast): +Infinity values cannot be represented ("
               << m << "); assuming infinitesimally large value." << std::endl;
      else if (warningcount == 0)
         trace
               << "WARNING (logrealfast): last warning repeated too many times; stopped logging." << std::endl;
#endif
      return -DBL_MAX;
      }
   // trap NaN
   else if (isnan(m))
      {
#ifndef NDEBUG
      static int warningcount = 10;
      if (--warningcount > 0)
         trace << "WARNING (logrealfast): NaN values cannot be represented ("
               << m << "); assuming infinitesimally small value." << std::endl;
      else if (warningcount == 0)
         trace
               << "WARNING (logrealfast): last warning repeated too many times; stopped logging." << std::endl;
#endif
      return DBL_MAX;
      }
   // trap negative numbers & zero
   else if (m <= 0)
      {
#ifndef NDEBUG
      static int warningcount = 10;
      if (--warningcount > 0)
         trace
               << "WARNING (logrealfast): Non-positive numbers cannot be represented ("
               << m << "); assuming infinitesimally small value." << std::endl;
      else if (warningcount == 0)
         trace
               << "WARNING (logrealfast): last warning repeated too many times; stopped logging." << std::endl;
#endif
      return DBL_MAX;
      }
   // finally convert (value must be ok)
   else
      return -log(m);
   }

// Input/Output Operations

std::ostream& operator<<(std::ostream& sout, const logrealfast& x)
   {
   using std::ios;

   const double lg = -x.logval / log(10.0);

   const ios::fmtflags flags = sout.flags();
   sout.setf(ios::fixed, ios::floatfield);
   sout << ::pow(10.0, lg - floor(lg));
   sout.setf(ios::showpos);
   sout << "e" << int(floor(lg));
   sout.flags(flags);

   return sout;
   }

std::istream& operator>>(std::istream& sin, logrealfast& x)
   {
   assertalways(sin.good());
   // get the number representation as a string
   using std::string;
   string sval;
   sin >> sval;
   // split into mantissa and exponent
   size_t pos = sval.find('e');
   double man;
   int exp;
   if (pos != string::npos)
      {
      man = atof(sval.substr(0, pos).c_str());
      exp = atoi(sval.substr(pos+1).c_str());
      }
   else
      {
      man = atof(sval.c_str());
      exp = 0;
      }
   // convert to logvalue
   x.logval = exp * log(10.0);
   x.logval += log(man);

   assertalways(sin.good());
   return sin;
   }

} // end namespace
