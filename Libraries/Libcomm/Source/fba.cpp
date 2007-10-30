#include "fba.h"     

namespace libcomm {

using libbase::matrix;
using libbase::vector;

// Initialization

template <class real, class dbl, class sig> void fba<real,dbl,sig>::init(const int N, const int q, const int I, const int xmax)
   {
   // code parameters
   //assert(n >= 1 && n <= 32);
   assert(N > 0);
   assert(q > 0);
   fba::N = N;
   fba::q = q;
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
   F = real(0);
   F(0, 0+N-1) = real(1);
   // we also know x[N] = equiprobable, since there will be no tail to speak of
   B = real(0);
   for(int y=-(N-1); y<=xmax; y++)
      B(N, y+N-1) = real(1);
   }

// Memory allocation

template <class real, class dbl, class sig> void fba<real,dbl,sig>::allocate()
   {
   // F & B need indices (j,y) where j in [0, N-1] ([0, N] for B) and y in [-(N-1), xmax]
   // to satisfy indexing requirements, instead of using y we use y+(N-1), which is in [0, xmax+N-1]
   F.init(N, xmax+N);
   B.init(N+1, xmax+N);

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

template <class real, class dbl, class sig> fba<real,dbl,sig>::fba(const int N, const int q, const int I, const int xmax)
   {
   init(N, q, I, xmax);
   }

template <class real, class dbl, class sig> fba<real,dbl,sig>::~fba()
   {
   }

   
// Internal procedures

template <class real, class dbl, class sig> void fba<real,dbl,sig>::work_forward(const vector<sig>& r)
   {
   for(int j=1; j<N; j++)
      for(int y=-j; y<=xmax; y++)
         {
         F(j,y+N-1) = 0;
         for(int a=y-I; a<=y+1 && y-a<=xmax; a++)
            F(j,y+N-1) += F(j-1,a+N-1) * real( P(a,y) * Q(a,y,j-1,r(j+y-1)) );
         }
   }

template <class real, class dbl, class sig> void fba<real,dbl,sig>::work_backward(const vector<sig>& r)
   {
   for(int j=N-1; j>=0; j--)
      for(int y=-j; y<=xmax; y++)
         {
         B(j,y+N-1) = 0;
         for(int b=y-1; b<=y+I && b-y<=xmax; b++)
            B(j,y+N-1) += B(j+1,b+N-1) * real( P(y,b) * Q(y,b,j+1,r(j+y+1)) );
         }
   }

// User procedures

template <class real, class dbl, class sig> void fba<real,dbl,sig>::decode(const vector<sig>& r, matrix<dbl>& p)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // compute forwards and backwards passes
   work_forward(r);
   work_backward(r);
   
   // compute results
   // Initialise result vector (one sparse symbol per timestep)
   p.init(N, q);
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

#define VERSION 1.10

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
