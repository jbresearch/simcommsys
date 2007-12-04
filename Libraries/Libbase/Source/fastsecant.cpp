#include "fastsecant.h"
#include <stdlib.h>
#include <math.h>

namespace libbase {

const vcs fastsecant::version("Semi-cached root-finding by Secant method module (fastsecant)", 1.10);

// exported functions

fastsecant::fastsecant(double (*func)(double)) : secant(func)
   {
   }
                        
void fastsecant::seed(const double x1, const double x2, const int n)
   {
   m_dMin = x1;
   m_dMax = x2;
   m_dStep = (x2-x1)/double(n-1);
   m_vdCache.init(n);
   double x = m_dMin;
   for(int i=0; i<n; i++)
      {
      m_vdCache(i) = secant::solve(x);
      x += m_dStep;
      }
   }
   
double fastsecant::solve(const double y)
   {
   const int i = int(floor((y-m_dMin)/m_dStep));
   const int j = int(ceil((y-m_dMin)/m_dStep));
   if(i == j)
      return m_vdCache(i);
   else if(i >= 0 && j < m_vdCache.size())
      {
      const double x1 = m_vdCache(i);
      const double x2 = m_vdCache(j);
      secant::seed(x1,x2);
      }
   else
      secant::seed(-1,1);

   return secant::solve(y);
   }

}; // end namespace
