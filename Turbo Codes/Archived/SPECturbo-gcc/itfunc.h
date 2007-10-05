#ifndef __itfunc_h
#define __itfunc_h

#include "config.h"

#include <math.h>
#include <iostream.h>
#include <stdlib.h>

inline double log2(const double x)
   {
   return log(x)/log(2);
   }

inline double gauss(const double x)
   {
   return exp(-0.5 * x * x)/sqrt(2.0 * PI);
   }

inline double Q(const double x)
   {
   const double sqrt_2 = 1.0 / sqrt(2.0);
   return 0.5 * erfc(x * sqrt_2);
   }
   
inline int weight(const int cw)
   {
   int c = cw;
   int w = 0;
   while(c)
      {
      w += (c & 1);
      c >>= 1;
      }
   return w;
   }

inline int factorial(const int x)
   {
   if(x < 0)
      {
      cerr << "FATAL ERROR: factorial range error (" << x << ")\n";
      exit(1);
      }
   int z = 1;
   for(int i=x; i>1; i--)
      z *= i;
   return z;
   }

inline int permutations(const int n, const int r)
   {
   if(n < r)
      {
      cerr << "FATAL ERROR: permutations range error (" << n << ", " << r << ")\n";
      exit(1);
      }
   int z = 1;
   for(int i=n; i>(n-r); i--)
      z *= i;
   return z;
   }

inline int combinations(const int n, const int r)
   {
   return permutations(n,r)/factorial(r);
   }

inline double factoriald(const int x)
   {
   if(x < 0)
      {
      cerr << "FATAL ERROR: factorial range error (" << x << ")\n";
      exit(1);
      }
   double z = 1;
   for(int i=x; i>1; i--)
      z *= i;
   return z;
   }

inline double permutationsd(const int n, const int r)
   {
   if(n < r)
      {
      cerr << "FATAL ERROR: permutations range error (" << n << ", " << r << ")\n";
      exit(1);
      }
   double z = 1;
   for(int i=n; i>(n-r); i--)
      z *= i;
   return z;
   }

inline double combinationsd(const int n, const int r)
   {
   return permutationsd(n,r)/factoriald(r);
   }

#endif

