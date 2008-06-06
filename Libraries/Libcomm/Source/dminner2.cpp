/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "dminner2.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// internally-used functions

/*!
   \brief Set up LUT with the lowest weight codewords
*/
template <class real> int dminner2<real>::fill(int i, libbase::bitfield suffix, int w)
   {
   // set up if this is the first (root) call
   if(i == 0 && w == -1)
      {
      assert(n >= 1 && n <= 32);
      assert(k >= 1 && k <= n);
      userspecified = false;
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

template <class real> void dminner2<real>::createsequence(const int tau)
   {
   // creates 'tau' elements of 'n' bits each
   ws.init(tau);
   for(int i=0; i<tau; i++)
      ws(i) = r.ival(1<<n);
   }

//! Inform user if I or xmax have changed

template <class real> void dminner2<real>::checkforchanges(int I, int xmax)
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

// initialization / de-allocation

template <class real> void dminner2<real>::init()
   {
   using libbase::bitfield;
   using libbase::weight;
   using libbase::trace;
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
   // Seed the watermark generator and clear the sequence
   r.seed(0);
   ws.init(0);
   // Clear bound channel
   mychan = NULL;
   }

template <class real> void dminner2<real>::free()
   {
   if(mychan != NULL)
      delete mychan;
   }

// constructor / destructor

template <class real> dminner2<real>::dminner2(const int n, const int k)
   {
   // code parameters
   dminner2::n = n;
   dminner2::k = k;
   // initialize everything else that depends on the above parameters
   fill();
   init();
   }

// implementations of channel-specific metrics for fba2

template <class real> real dminner2<real>::Q(int d, int i, const libbase::vector<bool>& r) const
   {
   // 'tx' is the vector of transmitted symbols that we're considering
   libbase::vector<bool> tx;
   tx.init(n);
   const int w = ws(i);
   const int s = lut(d);
   // NOTE: we transmit the low-order bits first
   for(int bit=0, t=s^w; bit<n; bit++, t >>= 1)
      tx(bit) = (t&1);
   // compute the conditional probability
   return mychan->receive(tx, r);
   }

// encoding and decoding functions

template <class real> void dminner2<real>::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx)
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
            libbase::trace << "DEBUG (dminner2::modulate): word " << i << "\t";
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
template <class real> void dminner2<real>::demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable)
   {
   using libbase::trace;
   // Inherit block size from last modulation step
   const int q = 1<<k;
   const int N = ws.size();
   const int tau = N*n;
   assert(N > 0);
   // Clone channel for access within Q()
   free();
   assertalways(mychan = dynamic_cast<bsid *> (chan.clone()));
   // Determine required FBA parameter values
   const double Pd = mychan->get_pd();
   const int I = bsid::compute_I(tau, Pd);
   const int xmax = bsid::compute_xmax(tau, Pd, I);
   const int dxmax = bsid::compute_xmax(n, Pd, bsid::compute_I(n, Pd));
   checkforchanges(I, xmax);
   // Initialize & perform forward-backward algorithm
   fba2<real,bool>::init(N, n, q, I, xmax, dxmax);
   fba2<real,bool>::prepare(rx);
   libbase::matrix<real> p;
   fba2<real,bool>::work_results(rx,p);
   // check for numerical underflow
   const real scale = p.max();
   assert(scale != real(0));
   // normalize and copy results
   p /= scale;
   ptable.init(N,q);
   for(int i=0; i<N; i++)
      for(int d=0; d<q; d++)
         ptable(i,d) = p(i,d);
   }

// description output

template <class real> std::string dminner2<real>::description() const
   {
   std::ostringstream sout;
   sout << "DM Inner Code (" << n << "/" << k << ", " << lutname << " codebook)";
   return sout.str();
   }

// object serialization - saving

template <class real> std::ostream& dminner2<real>::serialize(std::ostream& sout) const
   {
   sout << n << '\n';
   sout << k << '\n';
   sout << userspecified << '\n';
   if(userspecified)
      {
      sout << lutname << '\n';
      assert(lut.size() == num_symbols());
      for(int i=0; i<lut.size(); i++)
         sout << libbase::bitfield(lut(i),n) << '\n';
      }
   return sout;
   }

// object serialization - loading

template <class real> std::istream& dminner2<real>::serialize(std::istream& sin)
   {
   free();
   sin >> n;
   sin >> k;
   sin >> userspecified;
   if(userspecified)
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
   else
      fill();
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2<logrealfast>;
template <> const serializer dminner2<logrealfast>::shelper = serializer("modulator", "dminner2<logrealfast>", dminner2<logrealfast>::create);

}; // end namespace
