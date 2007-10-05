#include "secant.h"

#include <stdlib.h>
#include <math.h>

const vcs secant_version("Root-finding by Secant method module (secant)", 1.00);
                     
secant::secant(double (*func)(double))
   {
   bind(func);
   seed(0, 1);
   accuracy(1e-10);
   maxiter(1000);
   }
                        
void secant::bind(double (*func)(double))
   {
   f = func;
   }

void secant::seed(double x1, double x2)
   {
   init_x1 = x1;
   init_x2 = x2;
   }
   
void secant::accuracy(double dx)
   {
   min_dx = dx;
   }

void secant::maxiter(int n)
   {
   max_iter = n;
   }

double secant::solve(double y)
   {
   if(f == NULL)
      {
      cerr << "FATAL ERROR (secant): No function bound.\n";
      exit(1);
      }
   
   // Initialise
   double x1 = init_x1;
   double x2 = init_x2;
   double y1 = (*f)(x1) - y;
   double y2 = (*f)(x2) - y;
   
   if(fabs(y2) < fabs(y1))
      {
      swap(x1, x2);
      swap(y1, y2);
      }
   
   for(int i=0; i<max_iter; i++)
      {
      double dx = (x2-x1)*y1/(y1-y2);
      x2 = x1;
      y2 = y1;
      x1 += dx;
      y1 = (*f)(x1) - y;
      if(y1 == 0.0 || fabs(dx) < min_dx)
         return x1;
      }
   
   cerr << "FATAL ERROR (secant): Maximum number of iterations exceeded.\n";
   exit(1);
   }
   

