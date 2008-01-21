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
      if(tau > 32)
         std::cerr << libbase::pacifier("FBA Forward Pass", j-1, tau-1);
      // determine the strongest path at this point
      real threshold = 0;
      for(int a=-xmax; a<=xmax; a++)
         if(F(j-1,a) > threshold)
            threshold = F(j-1,a);
      threshold *= 1e-15;
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y < r.size()
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      const int amin = max(-xmax,1-j);
      const int amax = xmax;
      for(int a=amin; a<=amax; a++)
         {
         // ignore paths below a certain threshold
         if(F(j-1,a) < threshold)
            continue;
         const int ymin = max(-xmax,a-1);
         const int ymax = min(min(xmax,a+I),r.size()-j);
         for(int y=ymin; y<=ymax; y++)
            F(j,y) += F(j-1,a) * P(a,y) * Q(a,y,j-1,r.extract(j-1+a,y-a+1));
         }
      }
   if(tau > 32)
      std::cerr << libbase::pacifier("FBA Forward Pass", tau-1, tau-1);
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
   assertalways(abs(r.size()-tau) <= xmax);
   B(tau,r.size()-tau) = real(1);
   // compute remaining matrix values
   for(int i=0, j=tau-1; j>=0; i++, j--)
      {
      if(tau > 32)
         std::cerr << libbase::pacifier("FBA Backward Pass", i, tau);
      // determine the strongest path at this point
      real threshold = 0;
      for(int b=-xmax; b<=xmax; b++)
         if(B(j+1,b) > threshold)
            threshold = B(j+1,b);
      threshold *= 1e-15;
      // event must fit the received sequence:
      // 1. j+y >= 0
      // 2. j+b < r.size()
      // limits on insertions and deletions must be respected:
      // 3. b-y <= I
      // 4. b-y >= -1
      const int bmin = -xmax;
      const int bmax = min(xmax,r.size()-j-1);
      for(int b=bmin; b<=bmax; b++)
         {
         // ignore paths below a certain threshold
         if(B(j+1,b) < threshold)
            continue;
         const int ymin = max(max(-xmax,b-I),-j);
         const int ymax = min(xmax,b+1);
         for(int y=ymin; y<=ymax; y++)
            B(j,y) += B(j+1,b) * P(y,b) * Q(y,b,j,r.extract(j+y,b-y+1));
         }
      }
   if(tau > 32)
      std::cerr << libbase::pacifier("FBA Backward Pass", tau-1, tau-1);
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

template class fba<double>;
template class fba<mpreal>;
template class fba<mpgnu>;
template class fba<logreal>;
template class fba<logrealfast>;

}; // end namespace
