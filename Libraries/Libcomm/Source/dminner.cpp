/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "dminner.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// internally-used functions

/*!
   \brief Set up LUT with the lowest weight codewords
*/
template <class real, bool normalize>
int dminner<real,normalize>::fill(int i, libbase::bitfield suffix, int w)
   {
   assert(!user_lut);
   // set up if this is the first (root) call
   if(i == 0 && w == -1)
      {
      assert(n >= 1 && n <= 32);
      assert(k >= 1 && k <= n);
      user_lut = false;
      lutname = "sequential";
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

//! Watermark sequence creator

template <class real, bool normalize>
void dminner<real,normalize>::createsequence(const int tau)
   {
   // creates 'tau' elements of 'n' bits each
   ws.init(tau);
   for(int i=0; i<tau; i++)
      ws(i) = r.ival(1<<n);
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
void dminner<real,normalize>::set_blocksize(const channel<bool>& chan) const
   {
   const bsid& referred = dynamic_cast<const bsid &>(chan);
   referred.set_blocksize(n);
   }

template <class real, bool normalize>
void dminner<real,normalize>::work_results(const libbase::vector<bool>& r, libbase::matrix<real>& ptable, const int xmax, const int dxmax, const int I) const
   {
   // Inherit block size from last modulation step
   const int q = 1<<k;
   const int N = ws.size();
   // Initialise result vector (one sparse symbol per timestep)
   ptable.init(N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for(int i=0; i<N; i++)
      {
      std::cerr << libbase::pacifier("FBA Results", i, N);
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
         libbase::vector<bool> tx(n);
         for(int j=0, t=ws(i)^lut(d); j<n; j++, t >>= 1)
            tx(j) = (t&1);
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
            const real F = fba<real,bool,normalize>::getF(n*i,x1);
            // ignore paths below a certain threshold
            if(F < threshold)
               continue;
            const int x2min = std::max(-xmax,std::max(-n,-dxmax)+x1);
            const int x2max = std::min(std::min(xmax,std::min(n*I,dxmax)+x1),r.size()-n*(i+1));
            for(int x2=x2min; x2<=x2max; x2++)
               {
               // compute the conditional probability
               const real P = mychan->receive(tx, r.extract(n*i+x1,x2-x1+n));
               const real B = fba<real,bool,normalize>::getB(n*(i+1),x2);
               // include the probability for this particular sequence
               p += P * F * B;
               }
            }
         ptable(i,d) = p;
         }
      }
   if(N > 0)
      std::cerr << libbase::pacifier("FBA Results", N, N);
   }

// initialization / de-allocation

template <class real, bool normalize>
void dminner<real,normalize>::init()
   {
   using libbase::bitfield;
   using libbase::weight;
   using libbase::trace;
   // Fill default LUT if necessary
   if(!user_lut)
      fill();
#ifndef NDEBUG
   // Display LUT when debugging
   trace << "LUT (k=" << k << ", n=" << n << "):\n";
   for(int i=0; i<lut.size(); i++)
      trace << i << "\t" << bitfield(lut(i),n) << "\t" << weight(lut(i)) << "\n";
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
   libbase::vector<int> w = lut;
   w.apply(weight);
   f = w.sum()/double(n * w.size());
   trace << "Watermark code density = " << f << "\n";
   // set default thresholds if necessary
   if(!user_threshold)
      {
      th_inner = 1e-15;
      th_outer = 1e-6;
      }
   // Seed the watermark generator and clear the sequence
   r.seed(0);
   ws.init(0);
   // Clear bound channel
   mychan = NULL;
   }

template <class real, bool normalize>
void dminner<real,normalize>::free()
   {
   if(mychan != NULL)
      delete mychan;
   }

// constructor / destructor

template <class real, bool normalize>
dminner<real,normalize>::dminner(const int n, const int k)
   {
   // code parameters
   assert(k >= 1);
   assert(n > k);
   dminner::n = n;
   dminner::k = k;
   // default values
   user_lut = false;
   user_threshold = false;
   // initialize everything else that depends on the above parameters
   init();
   }

template <class real, bool normalize>
dminner<real,normalize>::dminner(const int n, const int k, const double th_inner, const double th_outer)
   {
   // code parameters
   assert(k >= 1);
   assert(n > k);
   dminner::n = n;
   dminner::k = k;
   // cutoff thresholds
   assert(th_inner <= 1);
   assert(th_outer <= 1);
   user_threshold = true;
   dminner::th_inner = th_inner;
   dminner::th_outer = th_outer;
   // default values
   user_lut = false;
   // initialize everything else that depends on the above parameters
   init();
   }

// implementations of channel-specific metrics for fba

template <class real, bool normalize>
real dminner<real,normalize>::P(const int a, const int b)
   {
   const int m = b-a;
   return Ptable[m];
   }

template <class real, bool normalize>
real dminner<real,normalize>::Q(const int a, const int b, const int i, const libbase::vector<bool>& s)
   {
   // 'a' and 'b' are redundant because 's' already contains the difference
   assert(s.size() == b-a+1);
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   const int word = i/n;
   const int bit  = i%n;
   bool tx = ((ws(word) >> bit) & 1);
   // compute the conditional probability
   return mychan->receive(tx, s);
   }

// encoding and decoding functions

template <class real, bool normalize>
void dminner<real,normalize>::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx)
   {
   // Inherit sizes
   const int q = 1<<k;
   const int tau = encoded.size();
   // We assume that each 'encoded' symbol can be fitted in an integral number of sparse vectors
   const int p = int(round(log(double(N))/log(double(q))));
   assert(N == pow(q, p));
   // Initialise result vector (one bit per sparse vector) and watermark sequence
   tx.init(n*p*tau);
   createsequence(p*tau);
   // Encode source stream
   for(int i=0, ii=0; i<tau; i++)
      for(int j=0, x=encoded(i); j<p; j++, ii++, x >>= k)
         {
         const int s = lut(x & (q-1));    // sparse vector
         const int w = ws(ii);            // watermark vector
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
            tx(ii*n+bit) = (t&1);
         }
   }

/*! \copydoc modulator::demodulate()

   \todo Make demodulation independent of the previous modulation step.
*/
template <class real, bool normalize>
void dminner<real,normalize>::demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable)
   {
   using libbase::trace;
   // Inherit block size from last modulation step
   const int q = 1<<k;
   const int N = ws.size();
   const int tau = N*n;
   assert(N > 0);
   // Set channel block size to q-ary symbol size
   set_blocksize(chan);
   // Clone channel for access within Q()
   free();
   assertalways(mychan = dynamic_cast<bsid *>(chan.clone()));
   // Update substitution probability to take into account sparse addition
   const double Ps = mychan->get_ps();
   mychan->set_ps(Ps*(1-f) + (1-Ps)*f);
   // Update block size to take into account the number of sparse symbols
   mychan->set_blocksize(tau);
   // Determine required FBA parameter values
   const double Pd = mychan->get_pd();
   const int I = bsid::compute_I(tau, Pd);
   const int xmax = bsid::compute_xmax(tau, Pd, I);
   const int dxmax = bsid::compute_xmax(n, Pd);
   checkforchanges(I, xmax);
   // Pre-compute 'P' table
   bsid::compute_Ptable(Ptable, xmax, mychan->get_pd(), mychan->get_pi());
   // Initialize & perform forward-backward algorithm
   fba<real,bool,normalize>::init(tau, I, xmax, th_inner);
   fba<real,bool,normalize>::prepare(rx);
   libbase::matrix<real> p;
   // Reset substitution probability to original value
   mychan->set_ps(Ps);
   work_results(rx,p,xmax,dxmax,I);
   // check for numerical underflow
   const real scale = p.max();
   assert(scale != real(0));
   // normalize and copy results
   p /= scale;
   ptable.init(N, q);
   for(int i=0; i<N; i++)
      for(int d=0; d<q; d++)
         ptable(i,d) = p(i,d);
   }

template <class real, bool normalize>
void dminner<real,normalize>::demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, const libbase::matrix<double>& app, libbase::matrix<double>& ptable)
   {
   }

// description output

template <class real, bool normalize>
std::string dminner<real,normalize>::description() const
   {
   std::ostringstream sout;
   sout << "DM Inner Code (" << n << "/" << k << ", " << lutname << " codebook";
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
   sout << user_lut << '\n';
   if(user_lut)
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
   free();
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
   sin >> user_lut;
   if(user_lut)
      {
      sin >> lutname;
      lut.init(num_symbols());
      libbase::bitfield b;
      for(int i=0; i<lut.size(); i++)
         {
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
   = serializer("modulator", "dminner<logrealfast>", dminner<logrealfast,false>::create);

template class dminner<double,true>;
template <>
const serializer dminner<double,true>::shelper \
   = serializer("modulator", "dminner<double>", dminner<double,true>::create);

}; // end namespace
