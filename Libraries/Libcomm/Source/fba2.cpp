/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "fba2.h"

namespace libcomm {

using libbase::matrix;
using libbase::vector;

// Memory allocation

template <class real, class sig> void fba2<real,sig>::allocate()
   {
   // determine limits
   dxmin = max(-n,-xmax);
   dxmax = min(n*I,xmax);
   // alpha needs indices (i,x) where i in [0, N-1] and x in [-xmax, xmax]
   // beta needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   // gamma needs indices (d,i,x,deltax) where d in [0, q-1], i in [0, N-1]
   // x in [-xmax, xmax], and deltax in [max(-n,-xmax), min(nI,xmax)]
   m_alpha.init(N, 2*xmax+1);
   m_beta.init(N+1, 2*xmax+1);
   m_gamma.init(q,N);
   for(int d=0; d<q; d++)
      for(int i=0; i<N; i++)
         m_gamma(d,i).init(2*xmax+1, dxmax-dxmin+1);
   // flag the state of the arrays
   initialised = true;
   }

// Initialization

template <class real, class sig> void fba2<real,sig>::init(int N, int n, int q, int I, int xmax)
   {
   // code parameters
   assert(N > 0);
   assert(n > 0);
   fba2::N = N;
   fba2::n = n;
   assert(q > 1);
   fba2::q = q;
   // decoder parameters
   assert(I > 0);
   assert(xmax > 0);
   fba2::I = I;
   fba2::xmax = xmax;
   // set flag as necessary
   initialised = false;
   }

// Internal procedures

template <class real, class sig> void fba2<real,sig>::work_gamma(const vector<sig>& r)
   {
   assert(initialised);
   // initialise array:
   for(int d=0; d<q; d++)
      for(int i=0; i<N; i++)
         m_gamma(d,i) = real(0);
   // compute remaining matrix values
   for(int i=0; i<N; i++)
      {
      std::cerr << libbase::pacifier("FBA Gamma", i, N);
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*i+x1 >= 0
      // 2. n*(i+1)-1+x2 < r.size()
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= xmax
      // 6. x2-x1 >= -xmax
      const int x1min = max(-xmax,-n*i);
      const int x1max = xmax;
      for(int x1=x1min; x1<=x1max; x1++)
         {
         const int x2min = max(-xmax,dxmin+x1);
         const int x2max = min(min(xmax,dxmax+x1),r.size()-n*(i+1));
         for(int x2=x2min; x2<=x2max; x2++)
            for(int d=0; d<q; d++)
               gamma(d,i,x1,x2-x1) = Q(d,i,r.extract(n*i+x1,n+x2-x1));
         }
      }
   std::cerr << libbase::pacifier("FBA Gamma", N, N);
   }

template <class real, class sig> void fba2<real,sig>::work_alpha(const vector<sig>& r)
   {
   assert(initialised);
   // initialise array:
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   m_alpha = real(0);
   alpha(0,0) = real(1);
   // compute remaining matrix values
   for(int i=1; i<N; i++)
      {
      std::cerr << libbase::pacifier("FBA Alpha", i-1, N-1);
      // determine the strongest path at this point
      real threshold = 0;
      for(int x1=-xmax; x1<=xmax; x1++)
         if(alpha(i-1,x1) > threshold)
            threshold = alpha(i-1,x1);
      threshold *= 1e-15;
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*(i-1)+x1 >= 0
      // 2. n*i-1+x2 < r.size()
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= xmax
      // 6. x2-x1 >= -xmax
      const int x1min = max(-xmax,-n*(i-1));
      const int x1max = xmax;
      for(int x1=x1min; x1<=x1max; x1++)
         {
         // ignore paths below a certain threshold
         if(alpha(i-1,x1) < threshold)
            continue;
         const int x2min = max(-xmax,dxmin+x1);
         const int x2max = min(min(xmax,dxmax+x1),r.size()-n*i);
         for(int x2=x2min; x2<=x2max; x2++)
            for(int d=0; d<q; d++)
               alpha(i,x2) += alpha(i-1,x1) * gamma(d,i-1,x1,x2-x1);
         }
      }
   std::cerr << libbase::pacifier("FBA Alpha", N-1, N-1);
   }

template <class real, class sig> void fba2<real,sig>::work_beta(const vector<sig>& r)
   {
   assert(initialised);
   // initialise array:
   // we also know x[tau] = r.size()-tau;
   // ie. drift before transmitting bit t[tau] is the discrepancy in the received vector size from tau
   const int tau = N*n;
   m_beta = real(0);
   assertalways(abs(r.size()-tau) <= xmax);
   beta(N,r.size()-tau) = real(1);
   // compute remaining matrix values
   for(int i=N-1; i>=0; i--)
      {
      std::cerr << libbase::pacifier("FBA Beta", N-1-i, N);
      // determine the strongest path at this point
      real threshold = 0;
      for(int x2=-xmax; x2<=xmax; x2++)
         if(beta(i+1,x2) > threshold)
            threshold = beta(i+1,x2);
      threshold *= 1e-15;
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*i+x1 >= 0
      // 2. n*(i+1)-1+x2 < r.size()
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= xmax
      // 6. x2-x1 >= -xmax
      const int x2min = -xmax;
      const int x2max = min(xmax,r.size()-n*(i+1));
      for(int x2=x2min; x2<=x2max; x2++)
         {
         // ignore paths below a certain threshold
         if(beta(i+1,x2) < threshold)
            continue;
         const int x1min = max(max(-xmax,x2-dxmax),-n*i);
         const int x1max = min(xmax,x2-dxmin);
         for(int x1=x1min; x1<=x1max; x1++)
            for(int d=0; d<q; d++)
               beta(i,x1) += beta(i+1,x2) * gamma(d,i,x1,x2-x1);
         }
      }
   std::cerr << libbase::pacifier("FBA Beta", N, N);
   }

// User procedures

template <class real, class sig> void fba2<real,sig>::prepare(const vector<sig>& r)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // compute forwards and backwards passes
   work_gamma(r);
   work_alpha(r);
   work_beta(r);
   }

template <class real, class sig> void fba2<real,sig>::work_results(const vector<sig>& r, libbase::matrix<real>& ptable) const
   {
   assert(initialised);
   // Initialise result vector (one sparse symbol per timestep)
   ptable.init(N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for(int i=0; i<N; i++)
      {
      std::cerr << libbase::pacifier("FBA Results", i, N);
      // determine the strongest path at this point
      real threshold = 0;
      for(int x1=-xmax; x1<=xmax; x1++)
         if(alpha(i,x1) > threshold)
            threshold = alpha(i,x1);
      threshold *= 1e-6;
      for(int d=0; d<q; d++)
         {
         real p = 0;
         // event must fit the received sequence:
         // (this is limited to start and end conditions)
         // 1. n*i+x1 >= 0
         // 2. n*(i+1)-1+x2 < r.size()
         // limits on insertions and deletions must be respected:
         // 3. x2-x1 <= n*I
         // 4. x2-x1 >= -n
         // limits on introduced drift in this section:
         // (necessary for forward recursion on extracted segment)
         // 5. x2-x1 <= xmax
         // 6. x2-x1 >= -xmax
         const int x1min = max(-xmax,-n*i);
         const int x1max = xmax;
         for(int x1=x1min; x1<=x1max; x1++)
            {
            // ignore paths below a certain threshold
            if(alpha(i,x1) < threshold)
               continue;
            const int x2min = max(-xmax,dxmin+x1);
            const int x2max = min(min(xmax,dxmax+x1),r.size()-n*(i+1));
            for(int x2=x2min; x2<=x2max; x2++)
               p += alpha(i,x1) * gamma(d,i,x1,x2-x1) * beta(i,x2);
            }
         ptable(i,d) = p;
         }
      }
   if(N > 0)
      std::cerr << libbase::pacifier("FBA Results", N, N);
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

template class fba2<double>;
template class fba2<logrealfast>;

template class fba2<double,bool>;
template class fba2<logrealfast,bool>;

}; // end namespace
