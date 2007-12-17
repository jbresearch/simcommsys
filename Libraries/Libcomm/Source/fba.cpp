/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "fba.h"     

namespace libcomm {

using libbase::matrix;
using libbase::vector;

// Initialization

template <class real, class sig> void fba<real,sig>::init(const int tau, const int I, const int xmax)
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

// Memory allocation

template <class real, class sig> void fba<real,sig>::allocate()
   {
   // F needs indices (j,y) where j in [0, tau-1] and y in [-xmax, xmax]
   // B needs indices (j,y) where j in [0, tau] and y in [-xmax, xmax]
   // to satisfy indexing requirements, instead of using y we use y+xmax, which is in [0, 2*xmax]
   mF.init(tau, 2*xmax+1);
   mB.init(tau+1, 2*xmax+1);

   // flag the state of the arrays
   initialised = true;
   }


// Creation/Destruction routines

template <class real, class sig> fba<real,sig>::fba()
   {
   initialised = false;
   }

template <class real, class sig> fba<real,sig>::fba(const int tau, const int I, const int xmax)
   {
   init(tau, I, xmax);
   }

template <class real, class sig> fba<real,sig>::~fba()
   {
   }

   
// Internal procedures

template <class real, class sig> void fba<real,sig>::work_forward(const vector<sig>& r)
   {
   using libbase::trace;
#ifndef NDEBUG
   if(tau > 32)
      trace << "DEBUG (fba): computing forward metrics...\n";
#endif

   // initialise memory if necessary
   if(!initialised)
      allocate();

   // initialise array:
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   mF = real(0);
   F(0,0) = real(1);
   // compute remaining matrix values
   for(int j=1; j<tau; j++)
      {
#ifndef NDEBUG
      if(tau > 32)
         trace << libbase::pacifier(100*(j-1)/(tau-1));
#endif
      // event must fit the received sequence - requirements:
      // 1. j-1+a >= 0
      // 2. j-1+y < r.size()
      const int ymin = -xmax;
      const int ymax = min(xmax,r.size()-j);
      for(int y=ymin; y<=ymax; y++)
         {
         F(j,y) = 0;
         const int amin = max(max(y-I,-xmax),1-j);
         const int amax = min(y+1,xmax);
         for(int a=amin; a<=amax; a++)
            F(j,y) += F(j-1,a) * P(a,y) * Q(a,y,j-1,r.extract(j-1+a,y-a+1));
         }
      }
#ifndef NDEBUG
   if(tau > 32)
      trace << "DEBUG (fba): forward metrics done.\n";
#endif
   }

template <class real, class sig> void fba<real,sig>::work_backward(const vector<sig>& r)
   {
   using libbase::trace;
#ifndef NDEBUG
   if(tau > 32)
      trace << "DEBUG (fba): computing backward metrics...\n";
#endif

   // initialise memory if necessary
   if(!initialised)
      allocate();

   // initialise array:
   // we also know x[tau] = r.size()-tau;
   // ie. drift before transmitting bit t[tau] is the discrepancy in the received vector size from tau
   mB = real(0);
   assert(abs(r.size()-tau) <= xmax);
   B(tau,r.size()-tau) = real(1);
   // compute remaining matrix values
   for(int j=tau-1; j>=0; j--)
      {
#ifndef NDEBUG
      if(tau > 32)
         trace << libbase::pacifier(100*(tau-2-j)/(tau-1));
#endif
      // event must fit the received sequence - requirements:
      // 1. j+y >= 0
      // 2. j+b < r.size()
      const int ymin = max(-xmax,-j);
      const int ymax = xmax;
      for(int y=ymin; y<=ymax; y++)
         {
         B(j,y) = 0;
         const int bmin = max(y-1,-xmax);
         const int bmax = min(min(y+I,xmax),r.size()-j-1);
         for(int b=bmin; b<=bmax; b++)
            B(j,y) += B(j+1,b) * P(y,b) * Q(y,b,j,r.extract(j+y,b-y+1));
         }
      }
#ifndef NDEBUG
   if(tau > 32)
      trace << "DEBUG (fba): backward metrics done.\n";
#endif
   }

// User procedures

template <class real, class sig> void fba<real,sig>::prepare(const vector<sig>& r)
   {
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

template class fba<double>;
template class fba<mpreal>;
template class fba<mpgnu>;
template class fba<logreal>;
template class fba<logrealfast>;

}; // end namespace
