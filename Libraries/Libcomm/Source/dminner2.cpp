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

// implementations of channel-specific metrics for fba2

template <class real>
real dminner2<real>::Q(int d, int i, const libbase::vector<bool>& r) const
   {
   const int n = dminner<real>::n;
   // 'tx' is the vector of transmitted symbols that we're considering
   libbase::vector<bool> tx;
   tx.init(n);
   const int w = dminner<real>::ws(i);
   const int s = dminner<real>::lut(d);
   // NOTE: we transmit the low-order bits first
   for(int bit=0, t=s^w; bit<n; bit++, t >>= 1)
      tx(bit) = (t&1);
   // compute the conditional probability
   return dminner<real>::mychan->receive(tx, r);
   }

// encoding and decoding functions

/*! \copydoc modulator::demodulate()

   \todo Make demodulation independent of the previous modulation step.
*/
template <class real>
void dminner2<real>::demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable)
   {
   using libbase::trace;
   const int n = dminner<real>::n;
   // Inherit block size from last modulation step
   const int q = 1<<dminner<real>::k;
   const int N = dminner<real>::ws.size();
   const int tau = N*n;
   assert(N > 0);
   // Set channel block size to q-ary symbol size
   const bsid& chanref = dynamic_cast<const bsid &>(chan);
   chanref.set_blocksize(n);
   // Clone channel for access within Q()
   dminner<real>::free();
   assertalways(dminner<real>::mychan = dynamic_cast<bsid *> (chan.clone()));
   // Determine required FBA parameter values
   const double Pd = dminner<real>::mychan->get_pd();
   const int I = bsid::compute_I(tau, Pd);
   const int xmax = bsid::compute_xmax(tau, Pd, I);
   const int dxmax = bsid::compute_xmax(n, Pd, bsid::compute_I(n, Pd));
   dminner<real>::checkforchanges(I, xmax);
   // Initialize & perform forward-backward algorithm
   fba2<real,bool>::init(N, n, q, I, xmax, dxmax, dminner<real>::th_inner, dminner<real>::th_outer);
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

template <class real>
std::string dminner2<real>::description() const
   {
   std::ostringstream sout;
   sout << "Alternative " << dminner<real>::description();
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& dminner2<real>::serialize(std::ostream& sout) const
   {
   return dminner<real>::serialize(sout);
   }

// object serialization - loading

template <class real>
std::istream& dminner2<real>::serialize(std::istream& sin)
   {
   return dminner<real>::serialize(sin);
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2<logrealfast>;
template <>
const serializer dminner2<logrealfast>::shelper = serializer("modulator", "dminner2<logrealfast>", dminner2<logrealfast>::create);

}; // end namespace
