#include "fba.h"     

namespace libcomm {

using libbase::matrix;
using libbase::vector;

// Initialization

template <class real, class dbl> void fba<real,dbl>::init(const int n, const int I)
   {
   // copy constants
   fba::n = n;
   fba::I = I;
   // set flag as necessary
   initialised = false;
   }

// Reset start- and end-state probabilities

template <class real, class dbl> void fba<real,dbl>::reset()
   {
   if(!initialised)
      {
      allocate();
      return;
      }

   }

// Memory allocation

template <class real, class dbl> void fba<real,dbl>::allocate()
   {
   // F & B need indices (j,y) where j in [0, n-1] and y in [-(n-1), (n-1)I]
   // to satisfy indexing requirements, instead of using y we use y+(n-1), which is in [0, (n-1)(I+1)]
   F.init(n, (n-1)*(I+1)+1);
   B.init(n, (n-1)*(I+1)+1);

   // flag the state of the arrays
   initialised = true;

   // set initial conditions
   reset();
   }


// Creation/Destruction routines

template <class real, class dbl> fba<real,dbl>::fba()
   {
   initialised = false;
   }

template <class real, class dbl> fba<real,dbl>::fba(const int n, const int I)
   {
   init(n, I);
   }

template <class real, class dbl> fba<real,dbl>::~fba()
   {
   }

   
// Internal procedures

template <class real, class dbl> void fba<real,dbl>::work_forward(const vector<int>& r)
   {
   for(int j=1; j<n; j++)
      for(int y=-j; y<=j*I; y++)
         {
         F(j,y+n-1) = 0;
         for(int a=y-I; a<=y+1; a++)
            F(j,y+n-1) += F(j-1,a+n-1) * P(a,y) * Q(a,y,j-1,r(j+y-1));
         }
   }

template <class real, class dbl> void fba<real,dbl>::work_backward(const vector<int>& r)
   {
   for(int j=n-2; j>=0; j--)
      for(int y=-j; y<=j*I; y++)
         {
         B(j,y+n-1) = 0;
         for(int b=y-1; b<=y+I; b++)
            B(j,y+n-1) += B(j+1,b+n-1) * P(y,b) * Q(y,b,j+1,r(j+y+1));
         }
   }


// User procedures

template <class real, class dbl> void fba<real,dbl>::decode(const vector<int>& r, matrix<dbl>& p)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // compute forwards and backwards passes
   work_forward(r);
   work_backward(r);
   
   // compute results
   // p(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
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

#define VERSION 1.00

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
