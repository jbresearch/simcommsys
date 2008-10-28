/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "dminner.h"
#include "timer.h"
#include "pacifier.h"
#include <sstream>

namespace libcomm {

// internally-used functions

/*!
   \brief Set up LUT with the lowest weight codewords
*/
template <class real, bool normalize>
int dminner<real,normalize>::fill(int i, libbase::bitfield suffix, int w)
   {
   assert(lut_type == lut_straight);
   // set up if this is the first (root) call
   if(i == 0 && w == -1)
      {
      lut.init(num_symbols());
      suffix = "";
      w = n;
      }
   // stop here if we've reached the end
   if(i >= lut.size())
      return i;
   // otherwise, it all depends on the weight we're considering
   using libbase::bitfield;
   using libbase::trace;
   bitfield b;
   trace << "Starting fill with:\t" << suffix << "\t" << w << "\n";
   if(w == 0)
      lut(i++) = suffix;
   else
      {
      w--;
      if(suffix.size() == 0)
         i = fill(i,suffix,w);
      for(b="1"; b.size()+suffix.size()+w <= n; b=b+bitfield("0"))
         i = fill(i,b+suffix,w);
      }
   return i;
   }

//! Inform user if I or xmax have changed

template <class real, bool normalize>
void dminner<real,normalize>::checkforchanges(int I, int xmax) const
   {
   static int last_I = 0;
   static int last_xmax = 0;
   if(last_I != I || last_xmax != xmax)
      {
      std::cerr << "Watermark Demodulation: I = " << I << ", xmax = " << xmax << ".\n";
      last_I = I;
      last_xmax = xmax;
      }
   }

template <class real, bool normalize>
void dminner<real,normalize>::work_results(const array1b_t& r, array2r_t& ptable, const int xmax, const int dxmax, const int I) const
   {
   libbase::pacifier progress("FBA Results");
   // determine limits
   const int dmin = std::max(-n,-dxmax);
   const int dmax = std::min(n*I,dxmax);
   // Inherit block size from last modulation step
   const int q = 1<<k;
   const int N = ws.size();
   // Initialise result vector (one sparse symbol per timestep)
   ptable.init(N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for(int i=0; i<N; i++)
      {
      std::cerr << progress.update(i, N);
      // determine the strongest path at this point
      real threshold = 0;
      for(int x1=-xmax; x1<=xmax; x1++)
         if(fba<real,bool,normalize>::getF(n*i,x1) > threshold)
            threshold = fba<real,bool,normalize>::getF(n*i,x1);
      threshold *= th_outer;
      for(int d=0; d<q; d++)
         {
         real p = 0;
         // create the considered transmitted sequence
         array1b_t t(n);
         for(int j=0, tval=ws(i)^lut(d); j<n; j++, tval >>= 1)
            t(j) = (tval&1);
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
         const int x2max_bnd = std::min(xmax,r.size()-n*(i+1));
         for(int x1=x1min; x1<=x1max; x1++)
            {
            const real F = fba<real,bool,normalize>::getF(n*i,x1);
            // ignore paths below a certain threshold
            if(F < threshold)
               continue;
            const int x2min = std::max(-xmax,dmin+x1);
            const int x2max = std::min(x2max_bnd,dmax+x1);
            for(int x2=x2min; x2<=x2max; x2++)
               {
               // compute the conditional probability
               const real R = mychan.receive(t, r.extract(n*i+x1,x2-x1+n));
               const real B = fba<real,bool,normalize>::getB(n*(i+1),x2);
               // include the probability for this particular sequence
               p += F * R * B;
               }
            }
         ptable(i,d) = p;
         }
      }
   if(N > 0)
      std::cerr << progress.update(N, N);
   }

template <class real, bool normalize>
void dminner<real,normalize>::normalize_results(const array2r_t& in, array2d_t& out) const
   {
   const int N = in.xsize();
   const int q = in.ysize();
   // check for numerical underflow
   const real scale = in.max();
   assert(scale != real(0));
   // normalize and copy results
   out.init(N,q);
   for(int i=0; i<N; i++)
      for(int d=0; d<q; d++)
         out(i,d) = in(i,d)/scale;
   }

// initialization / de-allocation

template <class real, bool normalize>
void dminner<real,normalize>::init()
   {
   // Fill default LUT if necessary
   if(lut_type == lut_straight)
      fill();
#ifndef NDEBUG
   // Display LUT when debugging
   libbase::trace << "LUT (k=" << k << ", n=" << n << "):\n";
   for(int i=0; i<lut.size(); i++)
      libbase::trace << i << "\t" << libbase::bitfield(lut(i),n) << "\t" << libbase::weight(lut(i)) << "\n";
#endif
   // Validate LUT
   assertalways(lut.size() == num_symbols());
   for(int i=0; i<lut.size(); i++)
      {
      // all entries should be within size
      assertalways(lut(i) >= 0 && lut(i) < (1<<n));
      // all entries should be distinct
      for(int j=0; j<i; j++)
         assertalways(lut(i) != lut(j));
      }
   // Compute the mean density
   array1i_t w = lut;
   w.apply(libbase::weight);
   f = w.sum()/double(n * w.size());
   libbase::trace << "Watermark code density = " << f << "\n";
   // set default thresholds if necessary
   if(!user_threshold)
      {
      th_inner = 1e-15;
      th_outer = 1e-6;
      }
   // Seed the watermark generator and clear the sequence
   r.seed(0);
   ws.init(0);
   // Check that everything makes sense
   test_invariant();
   }

// constructor / destructor

template <class real, bool normalize>
dminner<real,normalize>::dminner(const int n, const int k) :
   n(n), k(k), lut_type(lut_straight), user_threshold(false)
   {
   init();
   }

template <class real, bool normalize>
dminner<real,normalize>::dminner(const int n, const int k, const double th_inner, const double th_outer) :
   n(n), k(k), lut_type(lut_straight), user_threshold(true), th_inner(th_inner), th_outer(th_outer)
   {
   init();
   }

// Watermark-specific setup functions

template <class real, bool normalize>
void dminner<real,normalize>::set_thresholds(const double th_inner, const double th_outer)
   {
   user_threshold = true;
   dminner::th_inner = th_inner;
   dminner::th_outer = th_outer;
   test_invariant();
   }

// implementations of channel-specific metrics for fba

template <class real, bool normalize>
real dminner<real,normalize>::R(const int i, const array1b_t& r)
   {
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   const int word = i/n;
   const int bit  = i%n;
   bool t = ((ws(word) >> bit) & 1);
   // compute the conditional probability
   return mychan.receive(t, r);
   }

// block advance operation - update watermark sequence

template <class real, bool normalize>
void dminner<real,normalize>::advance() const
   {
   // Inherit sizes
   const int tau = this->input_block_size();
   // Initialize space
   ws.init(tau);
   // creates 'tau' elements of 'n' bits each
   for(int i=0; i<tau; i++)
      ws(i) = r.ival(1<<n);
   // Select a random alphabet
   if(lut_type == lut_random)
      {
      lut.init(num_symbols());
      for(int i=0; i<num_symbols(); i++)
         lut(i) = r.ival(1<<n);
      }
   }

// encoding and decoding functions

template <class real, bool normalize>
void dminner<real,normalize>::domodulate(const int N, const array1i_t& encoded, array1b_t& tx)
   {
   // TODO: when N is removed from the interface, rename 'tau' to 'N'
   // Inherit sizes
   const int q = 1<<k;
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == encoded.size());
   // Each 'encoded' symbol must be representable by a single sparse vector
   assertalways(N == q);
   // Initialise result vector (one bit per sparse vector)
   tx.init(n*tau);
   assertalways(ws.size() == tau);
   // Encode source stream
   for(int i=0; i<tau; i++)
      {
      const int s = lut(encoded(i) & (q-1)); // sparse vector
      const int w = ws(i);                   // watermark vector
#ifndef NDEBUG
      if(tau < 10)
         {
         libbase::trace << "DEBUG (dminner::modulate): word " << i << "\t";
         libbase::trace << "s = " << libbase::bitfield(s,n) << "\t";
         libbase::trace << "w = " << libbase::bitfield(w,n) << "\n";
         }
#endif
      // NOTE: we transmit the low-order bits first
      for(int bit=0, t=s^w; bit<n; bit++, t >>= 1)
         tx(i*n+bit) = (t&1);
      }
   }

template <class real, bool normalize>
void dminner<real,normalize>::dodemodulate(const channel<bool>& chan, const array1b_t& rx, array2d_t& ptable)
   {
   // Inherit sizes
   const int N = this->input_block_size();
   const int tau = N*n;
   assert(N > 0);
   // Copy channel for access within Q()
   mychan = dynamic_cast<const bsid&>(chan);
   // Update substitution probability to take into account sparse addition
   const double Ps = mychan.get_ps();
   mychan.set_ps(Ps*(1-f) + (1-Ps)*f);
   // Set block size for main forward-backward pass
   mychan.set_blocksize(tau);
   // Determine required FBA parameter values
   const double Pd = mychan.get_pd();
   const int I = bsid::compute_I(tau, Pd);
   const int xmax = bsid::compute_xmax(tau, Pd, I);
   const int dxmax = bsid::compute_xmax(n, Pd);
   checkforchanges(I, xmax);
   // Initialize & perform forward-backward algorithm
   fba<real,bool,normalize>::init(tau, I, xmax, th_inner);
   fba<real,bool,normalize>::prepare(rx);
   // Reset substitution probability to original value
   mychan.set_ps(Ps);
   // Set block size for results-computation pass to q-ary symbol size
   mychan.set_blocksize(n);
   // Compute and normalize results
   array2r_t p;
   work_results(rx,p,xmax,dxmax,I);
   normalize_results(p,ptable);
   }

template <class real, bool normalize>
void dminner<real,normalize>::dodemodulate(const channel<bool>& chan, const array1b_t& rx, const array2d_t& app, array2d_t& ptable)
   {
   // Apply standard demodulation
   dminner<real,normalize>::dodemodulate(chan, rx, ptable);
   // Multiply-in a-priori probabilities
   const int q = ptable.xsize();
   const int N = ptable.ysize();
   assert(app.xsize() == q);
   assert(app.ysize() == N);
   for(int i=0; i<N; i++)
      for(int d=0; d<q; d++)
         ptable(i,d) *= app(i,d);
   }

// description output

template <class real, bool normalize>
std::string dminner<real,normalize>::description() const
   {
   std::ostringstream sout;
   sout << "DM Inner Code (" << n << "/" << k << ", ";
   switch(lut_type)
      {
      case lut_straight:
         sout << "sequential codebook";
         break;

      case lut_user:
         sout << lutname << " codebook";
         break;

      case lut_random:
         sout << "random codebook";
         break;
      }
   if(user_threshold)
      sout << ", thresholds " << th_inner << "/" << th_outer;
   if(normalize)
      sout << ", normalized";
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real, bool normalize>
std::ostream& dminner<real,normalize>::serialize(std::ostream& sout) const
   {
   sout << user_threshold << '\n';
   if(user_threshold)
      {
      sout << th_inner << '\n';
      sout << th_outer << '\n';
      }
   sout << n << '\n';
   sout << k << '\n';
   sout << lut_type << '\n';
   if(lut_type == lut_user)
      {
      sout << lutname << '\n';
      assert(lut.size() == num_symbols());
      for(int i=0; i<lut.size(); i++)
         sout << libbase::bitfield(lut(i),n) << '\n';
      }
   return sout;
   }

// object serialization - loading

template <class real, bool normalize>
std::istream& dminner<real,normalize>::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   sin >> user_threshold;
   // deal with inexistent flag as 'false'
   if(sin.fail())
      {
      sin.clear();
      sin.seekg(start);
      user_threshold = false;
      }
   // read or set default thresholds
   if(user_threshold)
      {
      sin >> th_inner;
      sin >> th_outer;
      }
   sin >> n;
   sin >> k;
   int temp;
   sin >> temp;
   lut_type = (lut_t)temp;
   if(lut_type == lut_user)
      {
      sin >> lutname;
      lut.init(num_symbols());
      for(int i=0; i<lut.size(); i++)
         {
         libbase::bitfield b;
         sin >> b;
         lut(i) = b;
         assertalways(n == b.size());
         }
      }
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner<logrealfast,false>;
template <>
const serializer dminner<logrealfast,false>::shelper \
   = serializer("blockmodem", "dminner<logrealfast>", dminner<logrealfast,false>::create);

template class dminner<double,true>;
template <>
const serializer dminner<double,true>::shelper \
   = serializer("blockmodem", "dminner<double>", dminner<double,true>::create);

}; // end namespace
