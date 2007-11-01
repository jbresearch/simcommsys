#include "fba.h"     

namespace libcomm {

using libbase::matrix;
using libbase::vector;

// Initialization

template <class real, class dbl, class sig> void fba<real,dbl,sig>::init(const int tau, const int I, const int xmax)
   {
   // code parameters
   assert(tau > 0);
   fba::tau = tau;
   // decoder parameters
   assert(I > 0);
   assert(xmax > 0);
   fba::I = I;
   fba::xmax = xmax;
   // set flag as necessary
   initialised = false;
   }

// Reset start- and end-state probabilities

template <class real, class dbl, class sig> void fba<real,dbl,sig>::reset()
   {
   if(!initialised)
      {
      allocate();
      return;
      }

   // initialise F and B arrays:
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   mF = real(0);
   F(0, 0) = real(1);
   // we also know x[tau] = equiprobable, since there will be no tail to speak of
   mB = real(0);
   for(int y=-(tau-1); y<=xmax; y++)
      B(tau, y) = real(1);
   }

// Memory allocation

template <class real, class dbl, class sig> void fba<real,dbl,sig>::allocate()
   {
   // F & B need indices (j,y) where j in [0, tau-1] ([0, tau] for B) and y in [-(tau-1), xmax]
   // to satisfy indexing requirements, instead of using y we use y+(tau-1), which is in [0, xmax+tau-1]
   mF.init(tau, xmax+tau);
   mB.init(tau+1, xmax+tau);

   // flag the state of the arrays
   initialised = true;

   // set initial conditions
   reset();
   }


// Creation/Destruction routines

template <class real, class dbl, class sig> fba<real,dbl,sig>::fba()
   {
   initialised = false;
   }

template <class real, class dbl, class sig> fba<real,dbl,sig>::fba(const int tau, const int I, const int xmax)
   {
   init(tau, I, xmax);
   }

template <class real, class dbl, class sig> fba<real,dbl,sig>::~fba()
   {
   }

   
// Internal procedures

template <class real, class dbl, class sig> void fba<real,dbl,sig>::work_forward(const vector<sig>& r)
   {
   for(int j=1; j<tau; j++)
      for(int y=-j; y<=xmax; y++)
         {
         F(j,y) = 0;
         for(int a=y-I; a<=y+1 && y-a<=xmax; a++)
            {
            vector<sig> s(y-a+1);
            for(int i=j-1+a, k=0; i<=j-1+y; i++, k++)
               s(k) = r(i);
            F(j,y) += F(j-1,a) * real( P(a,y) * Q(a,y,j-1,s) );
            }
         }
   }

template <class real, class dbl, class sig> void fba<real,dbl,sig>::work_backward(const vector<sig>& r)
   {
   for(int j=tau-1; j>=0; j--)
      for(int y=-j; y<=xmax; y++)
         {
         B(j,y) = 0;
         for(int b=y-1; b<=y+I && b-y<=xmax; b++)
            {
            vector<sig> s(b-y+1);
            for(int i=j+1+y, k=0; i<=j+1+b; i++, k++)
               s(k) = r(i);
            B(j,y) += B(j+1,b) * real( P(y,b) * Q(y,b,j+1,s) );
            }
         }
   }

// User procedures

template <class real, class dbl, class sig> void fba<real,dbl,sig>::prepare(const vector<sig>& r)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // compute forwards and backwards passes
   work_forward(r);
   work_backward(r);
   }

}; // end namespace

// Explicit Realizations

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

using libbase::vcs;

#define VERSION 1.20

template class fba<double>;
template <> const vcs fba<double>::version = vcs("Forward-Backward Algorithm module (fba<double>)", VERSION);

template class fba<mpreal>;
template <> const vcs fba<mpreal>::version = vcs("Forward-Backward Algorithm module (fba<mpreal>)", VERSION);

template class fba<mpgnu>;
template <> const vcs fba<mpgnu>::version = vcs("Forward-Backward Algorithm module (fba<mpgnu>)", VERSION);

template class fba<logreal>;
template <> const vcs fba<logreal>::version = vcs("Forward-Backward Algorithm module (fba<logreal>)", VERSION);

template class fba<logrealfast>;
template <> const vcs fba<logrealfast>::version = vcs("Forward-Backward Algorithm module (fba<logrealfast>)", VERSION);

}; // end namespace
