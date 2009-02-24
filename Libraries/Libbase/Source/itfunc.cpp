/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "itfunc.h"
#include <iostream>
#include <sstream>

namespace libbase {

using std::cerr;
using std::string;

double gammln(double xx)
   {
   double x,y,tmp,ser;
   double cof[6] = {76.18009172947146,-86.50532032941677,
      24.01409824083091,-1.231739572450155,
      0.1208650973866179e-2,-0.5395239384953e-5};

   y = x = xx;
   tmp = x+5.5;
   tmp -= (x+0.5)*log(tmp);
   ser = 1.000000000190015;
   for(int j=0; j<=5; j++)
      ser += cof[j]/++y;
   return -tmp+log(2.5066282746310005*ser/x);
   }

double gammser(double a, double x)
   {
   double gln = gammln(a);
   if(x < 0.0)
      {
      cerr << "FATAL ERROR: gammser range error (" << x << ")\n";
      exit(1);
      }
   else if(x == 0.0)
      return 0.0;
   else
      {
      int itmax = 100;
      double eps = 3.0e-7;
      double sum,del,ap;
      ap = a;
      del = sum = 1.0/a;
      for(int n=1; n<=itmax; n++)
         {
         ++ap;
         del *= x/ap;
         sum += del;
         if(fabs(del) < fabs(sum)*eps)
            return sum*exp(-x+a*log(x)-gln);
         }
      cerr << "FATAL ERROR: gammser error - a too large, itmax too small (" << a << ")\n";
      exit(1);
      }
   }

double gammcf(double a, double x)
   {
   int itmax = 100;
   double eps = 3.0e-7;
   double fpmin = 1.0e-30;
   double gln = gammln(a);

   double an,b,c,d,del,h;
   int i;

   b = x+1.0-a;
   c = 1.0/fpmin;
   d = 1.0/b;
   h = d;

   for(i=1; i<itmax; i++)
      {
      an = -i*(i-a);
      b += 2.0;
      d = an*d+b;
      if(fabs(d) < fpmin)
         d = fpmin;
      c = b+an/c;
      if(fabs(c) < fpmin)
         c = fpmin;
      d = 1.0/d;
      del = d*c;
      h *= del;
      if(fabs(del-1.0) < eps)
         break;
      }
   if(i > itmax)
      {
      cerr << "FATAL ERROR: gammcf error - a too large, itmax too small (" << a << ")\n";
      exit(1);
      }
   return exp(-x+a*log(x)-gln)*h;
   }

double gammp(double a, double x)
   {
   if(x < 0.0 || a <= 0.0)
      {
      cerr << "FATAL ERROR: gammp range error (" << a << ", " << x << ")\n";
      exit(1);
      }
   if(x < a+1.0)  // use series representation
      return gammser(a,x);
   else           // use continued fraction representation and take complement
      return 1.0-gammcf(a,x);
   }

/*! \brief Error function based on Chebychev fitting
   \sa Numerical Recipes in C, p.220
   Based on Chebychev fitting to an inspired guess as to the functional form.
   \note fractional error is everywhere less than 1.2E-7
   erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt
*/
double cerf(double x)
   {
   double z = fabs(x);
   double t = 1.0/(1.0+0.5*z);
   double ans = t*exp(-z*z - 1.26551223 + \
      t*(1.00002368 + t*(0.37409196 + t*(0.09678418 + t*(-0.18628806 + t*(0.27886807 + \
      t*(-1.13520398 + t*(1.48851587 + t*(-0.82215223 + t*0.17087277))) ))))) );
   return x >= 0.0 ? 1.0-ans : ans-1.0;
   }

/*! \brief Complementary Error function based on Chebychev fitting
   \sa Numerical Recipes in C, p.220
   Based on Chebychev fitting to an inspired guess as to the functional form.
   \note fractional error is everywhere less than 1.2E-7
   erfc(x) = 1-erf(x) = 2/sqrt(pi) * integral from x to inf of exp(-t^2) dt
*/
double cerfc(double x)
   {
   double z = fabs(x);
   double t = 1.0/(1.0+0.5*z);
   double ans = t*exp(-z*z - 1.26551223 + \
      t*(1.00002368 + t*(0.37409196 + t*(0.09678418 + t*(-0.18628806 + t*(0.27886807 + \
      t*(-1.13520398 + t*(1.48851587 + t*(-0.82215223 + t*0.17087277))) ))))) );
   return x >= 0.0 ? ans : 2.0-ans;
   }

/*! \brief Binary Hamming weight
*/
int weight(int cw)
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

/*! \brief Inverse Gray code
*/
int32u igray(int32u n)
   {
   int32u r = n;
   for(int i=1; i<32; i<<=1)
      r ^= r >> i;
   return r;
   }

/*! \brief Greatest common divisor
   GCD function based on Euclid's algorithm.
*/
int gcd(int a, int b)
   {
   while(b != 0)
      {
      int t = b;
      b = a % b;
      a = t;
      }
   return a;
   }

// combinatorial statistics functions

int factorial(int x)
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

int permutations(int n, int r)
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

double factoriald(int x)
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

double permutationsd(int n, int r)
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

/*! \brief Converts a string to its hex representation
*/
string hexify(const string input)
   {
   std::ostringstream sout;
   sout << std::hex;
   for(size_t i=0; i<input.length(); i++)
      {
      sout.width(2);
      sout.fill('0');
      sout << int(int8u(input.at(i)));
      }
   string output = sout.str();
   //trace << "(itfunc) hexify: (" << input << ") = " << output << ", length = " << output.length() << "\n";
   return output;
   }

/*! \brief Reconstructs a string from its hex representation
*/
string dehexify(const string input)
   {
   string output;
   for(size_t i=0; i<input.length(); i+=2)
      {
      string s = input.substr(i,2);
      if(s.length() == 1)
         s += '0';
      output += char(strtoul(s.c_str(),NULL,16));
      }
   //trace << "(itfunc) dehexify: " << input << " = (" << output << "), length = " << output.length() << "\n";
   return output;
   }

}; // end namespace
