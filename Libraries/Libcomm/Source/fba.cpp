/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "fba.h"
#include <iomanip>

namespace libcomm {

/*! \brief Memory allocator for working matrices
*/
template <class real, class sig, bool normalize>
void fba<real,sig,normalize>::allocate()
   {
   // Allocate required size
   // F needs indices (j,y) where j in [0, tau-1] and y in [-xmax, xmax]
   // B needs indices (j,y) where j in [1, tau] and y in [-xmax, xmax]
   typedef boost::multi_array_types::extent_range range;
   F.resize(boost::extents[tau][range(-xmax,xmax+1)]);
   B.resize(boost::extents[range(1,tau+1)][range(-xmax,xmax+1)]);
   // determine memory occupied and tell user
   std::ios::fmtflags flags = std::cerr.flags();
   std::cerr << "FBA Memory Usage: " << std::fixed << std::setprecision(1);
   std::cerr << sizeof(real)*( F.num_elements() + B.num_elements() )/double(1<<20) << "MB\n";
   std::cerr.setf(flags);
   // flag the state of the arrays
   initialised = true;
   }

// Initialization

template <class real, class sig, bool normalize>
void fba<real,sig,normalize>::init(int tau, int I, int xmax, double th_inner)
   {
   // code parameters
   assert(tau > 0);
   fba::tau = tau;
   // decoder parameters
   assert(I > 0);
   assert(xmax > 0);
   fba::I = I;
   fba::xmax = xmax;
   // path truncation parameters
   assert(th_inner >= 0);
   fba::th_inner = th_inner;
   // set flag as necessary
   initialised = false;
   }

// Internal procedures

template <class real, class sig, bool normalize>
void fba<real,sig,normalize>::work_forward(const array1s_t& r)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // initialise array:
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   typedef typename array2r_t::index index;
   F = real(0);
   F[0][0] = real(1);
   // compute remaining matrix values
   for(index j=1; j<tau; ++j)
      {
      std::cerr << libbase::pacifier("FBA Forward Pass", int(j-1), tau-1);
      // determine the strongest path at this point
      real threshold = 0;
      for(index a=-xmax; a<=xmax; ++a)
         if(F[j-1][a] > threshold)
            threshold = F[j-1][a];
      threshold *= th_inner;
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      const index amin = std::max(-xmax,1-int(j));
      const index amax = xmax;
      const index ymax_bnd = std::min(xmax,r.size()-int(j));
      for(index a=amin; a<=amax; ++a)
         {
         // ignore paths below a certain threshold
         if(F[j-1][a] < threshold)
            continue;
         const index ymin = std::max(-xmax,int(a)-1);
         const index ymax = std::min(ymax_bnd,int(a)+I);
         for(index y=ymin; y<=ymax; ++y)
            F[j][y] += F[j-1][a] * R(int(j-1),r.extract(int(j-1+a),int(y-a+1)));
         }
      // normalize if requested
      if(normalize)
         {
         real sum = 0;
         for(index y=-xmax; y<=xmax; ++y)
            sum += F[j][y];
         sum = real(1)/sum;
         for(index y=-xmax; y<=xmax; ++y)
            F[j][y] *= sum;
         }
      }
   std::cerr << libbase::pacifier("FBA Forward Pass", tau-1, tau-1);
   }

template <class real, class sig, bool normalize>
void fba<real,sig,normalize>::work_backward(const array1s_t& r)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // initialise array:
   // we also know x[tau] = r.size()-tau;
   // ie. drift before transmitting bit t[tau] is the discrepancy in the received vector size from tau
   typedef typename array2r_t::index index;
   B = real(0);
   assertalways(abs(r.size()-tau) <= xmax);
   B[tau][r.size()-tau] = real(1);
   // compute remaining matrix values
   for(index j=tau-1; j>0; --j)
      {
      std::cerr << libbase::pacifier("FBA Backward Pass", int(tau-1-j), tau-1);
      // determine the strongest path at this point
      real threshold = 0;
      for(index b=-xmax; b<=xmax; ++b)
         if(B[j+1][b] > threshold)
            threshold = B[j+1][b];
      threshold *= th_inner;
      // event must fit the received sequence:
      // 1. j+y >= 0
      // 2. j+b <= r.size()-1
      // limits on insertions and deletions must be respected:
      // 3. b-y <= I
      // 4. b-y >= -1
      const index bmin = -xmax;
      const index bmax = std::min(xmax,r.size()-int(j)-1);
      const index ymin_bnd = std::max(-xmax,int(-j));
      for(index b=bmin; b<=bmax; ++b)
         {
         // ignore paths below a certain threshold
         if(B[j+1][b] < threshold)
            continue;
         const index ymin = std::max(ymin_bnd,int(b)-I);
         const index ymax = std::min(xmax,int(b)+1);
         for(index y=ymin; y<=ymax; ++y)
            B[j][y] += B[j+1][b] * R(int(j),r.extract(int(j+y),int(b-y+1)));
         }
      // normalize if requested
      if(normalize)
         {
         real sum = 0;
         for(index y=-xmax; y<=xmax; ++y)
            sum += B[j][y];
         sum = real(1)/sum;
         for(index y=-xmax; y<=xmax; ++y)
            B[j][y] *= sum;
         }
      }
   std::cerr << libbase::pacifier("FBA Backward Pass", tau-1, tau-1);
   }

// User procedures

template <class real, class sig, bool normalize>
void fba<real,sig,normalize>::prepare(const array1s_t& r)
   {
   // compute forwards and backwards passes
   work_forward(r);
   work_backward(r);
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

template class fba<double,bool,true>;
template class fba<logrealfast,bool,false>;

}; // end namespace
