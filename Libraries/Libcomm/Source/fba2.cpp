/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "fba2.h"
#include <iomanip>

namespace libcomm {

// Memory allocation

/*! \brief Memory allocator for working matrices
*/
template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::allocate()
   {
   // determine limits
   dmin = std::max(-n,-dxmax);
   dmax = std::min(n*I,dxmax);
   // alpha needs indices (i,x) where i in [0, N-1] and x in [-xmax, xmax]
   // beta needs indices (i,x) where i in [1, N] and x in [-xmax, xmax]
   typedef boost::multi_array_types::extent_range range;
   alpha.resize(boost::extents[N][range(-xmax,xmax+1)]);
   beta.resize(boost::extents[range(1,N+1)][range(-xmax,xmax+1)]);
   // dynamically decide whether we want to use the gamma cache or not
   // decision is hardwired: use if memory requirement < 750MB
   cache_enabled = sizeof(real)*(q * N * (2*xmax+1) * (dmax-dmin+1)) < (750<<20);
   // gamma needs indices (d,i,x,deltax) where d in [0, q-1], i in [0, N-1]
   // x in [-xmax, xmax], and deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
   if(cache_enabled)
      {
      gamma.resize(boost::extents[q][N][range(-xmax,xmax+1)][range(dmin,dmax+1)]);
      cached.resize(boost::extents[N][range(-xmax,xmax+1)][range(dmin,dmax+1)]);
      }
   else
      {
      gamma.resize(boost::extents[0][0][0][0]);
      cached.resize(boost::extents[0][0][0]);
      std::cerr << "FBA Cache Disabled.\n";
      }
   // determine memory occupied and tell user
   std::ios::fmtflags flags = std::cerr.flags();
   std::cerr << "FBA Memory Usage: " << std::fixed << std::setprecision(1);
   std::cerr << ( sizeof(bool)*cached.num_elements() + sizeof(real)*
      ( alpha.num_elements() + beta.num_elements() + gamma.num_elements() )
      )/double(1<<20) << "MB\n";
   std::cerr.setf(flags);
   // flag the state of the arrays
   initialised = true;
   }

/*! \brief Release memory for working matrices
*/
template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::free()
   {
   alpha.resize(boost::extents[0][0]);
   beta.resize(boost::extents[0][0]);
   cache_enabled = false;
   gamma.resize(boost::extents[0][0][0][0]);
   cached.resize(boost::extents[0][0][0]);
   // flag the state of the arrays
   initialised = true;
   }

// Initialization

template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner, double th_outer)
   {
   // if any parameters that effect memory have changed, release memory
   if(initialised && (N != fba2::N || n != fba2::n || q != fba2::q
      || I != fba2::I || xmax != fba2::xmax || dxmax != fba2::dxmax))
      free();
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
   assert(dxmax > 0);
   fba2::I = I;
   fba2::xmax = xmax;
   fba2::dxmax = dxmax;
   // path truncation parameters
   assert(th_inner >= 0);
   assert(th_outer >= 0);
   fba2::th_inner = th_inner;
   fba2::th_outer = th_outer;
   }

// Internal procedures

template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::work_gamma(const array1s_t& r)
   {
   assert(initialised);
   if(!cache_enabled)
      return;
   // initialise array
   gamma = real(0);
   // initialize cache
   cached = false;
   }

template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::work_alpha(const array1s_t& r)
   {
   assert(initialised);
   // initialise array:
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   alpha = real(0);
   alpha[0][0] = real(1);
   // compute remaining matrix values
   for(int i=1; i<N; i++)
      {
      std::cerr << libbase::pacifier("FBA Alpha", i-1, N-1);
      // determine the strongest path at this point
      real threshold = 0;
      for(int x1=-xmax; x1<=xmax; x1++)
         if(alpha[i-1][x1] > threshold)
            threshold = alpha[i-1][x1];
      threshold *= th_inner;
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*(i-1)+x1 >= 0
      // 2. n*i-1+x2 <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= dxmax
      // 6. x2-x1 >= -dxmax
      const int x1min = std::max(-xmax,-n*(i-1));
      const int x1max = xmax;
      for(int x1=x1min; x1<=x1max; x1++)
         {
         // ignore paths below a certain threshold
         if(alpha[i-1][x1] < threshold)
            continue;
         const int x2min = std::max(-xmax,dmin+x1);
         const int x2max = std::min(std::min(xmax,dmax+x1),r.size()-n*i);
         for(int x2=x2min; x2<=x2max; x2++)
            for(int d=0; d<q; d++)
               alpha[i][x2] += alpha[i-1][x1] * compute_gamma(d,i-1,x1,x2-x1,r);
         }
      // normalize if requested
      if(normalize)
         {
         real sum = 0;
         for(int x=-xmax; x<=xmax; x++)
            sum += alpha[i][x];
         sum = real(1)/sum;
         for(int x=-xmax; x<=xmax; x++)
            alpha[i][x] *= sum;
         }
      }
   std::cerr << libbase::pacifier("FBA Alpha", N-1, N-1);
   }

template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::work_beta(const array1s_t& r)
   {
   assert(initialised);
   // initialise array:
   // we also know x[tau] = r.size()-tau;
   // ie. drift before transmitting bit t[tau] is the discrepancy in the received vector size from tau
   const int tau = N*n;
   beta = real(0);
   assertalways(abs(r.size()-tau) <= xmax);
   beta[N][r.size()-tau] = real(1);
   // compute remaining matrix values
   for(int i=N-1; i>0; i--)
      {
      std::cerr << libbase::pacifier("FBA Beta", N-1-i, N-1);
      // determine the strongest path at this point
      real threshold = 0;
      for(int x2=-xmax; x2<=xmax; x2++)
         if(beta[i+1][x2] > threshold)
            threshold = beta[i+1][x2];
      threshold *= th_inner;
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*i+x1 >= 0
      // 2. n*(i+1)-1+x2 <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= dxmax
      // 6. x2-x1 >= -dxmax
      const int x2min = -xmax;
      const int x2max = std::min(xmax,r.size()-n*(i+1));
      for(int x2=x2min; x2<=x2max; x2++)
         {
         // ignore paths below a certain threshold
         if(beta[i+1][x2] < threshold)
            continue;
         const int x1min = std::max(std::max(-xmax,x2-dmax),-n*i);
         const int x1max = std::min(xmax,x2-dmin);
         for(int x1=x1min; x1<=x1max; x1++)
            for(int d=0; d<q; d++)
               beta[i][x1] += beta[i+1][x2] * compute_gamma(d,i,x1,x2-x1,r);
         }
      // normalize if requested
      if(normalize)
         {
         real sum = 0;
         for(int x=-xmax; x<=xmax; x++)
            sum += beta[i][x];
         sum = real(1)/sum;
         for(int x=-xmax; x<=xmax; x++)
            beta[i][x] *= sum;
         }
      }
   std::cerr << libbase::pacifier("FBA Beta", N-1, N-1);
   }

// User procedures

template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::prepare(const array1s_t& r)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

#ifndef NDEBUG
   // reset cache counters
   gamma_calls = 0;
   gamma_misses = 0;
#endif

   // compute forwards and backwards passes
   work_gamma(r);
   work_alpha(r);
   work_beta(r);
   }

template <class real, class sig, bool normalize>
void fba2<real,sig,normalize>::work_results(const array1s_t& r, array2r_old_t& ptable) const
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
         if(alpha[i][x1] > threshold)
            threshold = alpha[i][x1];
      threshold *= th_outer;
      for(int d=0; d<q; d++)
         {
         real p = 0;
         // event must fit the received sequence:
         // (this is limited to start and end conditions)
         // 1. n*i+x1 >= 0
         // 2. n*(i+1)-1+x2 <= r.size()-1
         // limits on insertions and deletions must be respected:
         // 3. x2-x1 <= n*I
         // 4. x2-x1 >= -n
         // limits on introduced drift in this section:
         // (necessary for forward recursion on extracted segment)
         // 5. x2-x1 <= dxmax
         // 6. x2-x1 >= -dxmax
         const int x1min = std::max(-xmax,-n*i);
         const int x1max = xmax;
         for(int x1=x1min; x1<=x1max; x1++)
            {
            // ignore paths below a certain threshold
            if(alpha[i][x1] < threshold)
               continue;
            const int x2min = std::max(-xmax,dmin+x1);
            const int x2max = std::min(std::min(xmax,dmax+x1),r.size()-n*(i+1));
            for(int x2=x2min; x2<=x2max; x2++)
               p += alpha[i][x1] * compute_gamma(d,i,x1,x2-x1,r) * beta[i+1][x2];
            }
         ptable(i,d) = p;
         }
      }
   if(N > 0)
      std::cerr << libbase::pacifier("FBA Results", N, N);
#ifndef NDEBUG
   // show cache statistics
   std::cerr << "FBA Cache Usage: " << 100*gamma_misses/double(cached.num_elements()) << "%\n";
   std::cerr << "FBA Cache Reuse: " << gamma_calls/double(gamma_misses*q) << "x\n";
#endif
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

template class fba2<double,bool,true>;
template class fba2<logrealfast,bool,false>;

}; // end namespace
