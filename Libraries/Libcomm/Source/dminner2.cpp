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

template <class real, bool normalize>
real dminner2<real,normalize>::Q(int d, int i, const libbase::vector<bool>& r) const
   {
   const int n = dminner<real,normalize>::n;
   // 'tx' is the vector of transmitted symbols that we're considering
   libbase::vector<bool> tx;
   tx.init(n);
   const int w = dminner<real,normalize>::ws(i);
   const int s = dminner<real,normalize>::lut(d);
   // NOTE: we transmit the low-order bits first
   for(int bit=0, t=s^w; bit<n; bit++, t >>= 1)
      tx(bit) = (t&1);
   // compute the conditional probability
   return dminner<real,normalize>::mychan.receive(tx, r);
   }

// encoding and decoding functions

/*! \copydoc modulator::demodulate()

   \todo Make demodulation independent of the previous modulation step.
*/
template <class real, bool normalize>
void dminner2<real,normalize>::demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable)
   {
   // Inherit block size from last modulation step
   const int q = 1<<dminner<real,normalize>::k;
   const int n = dminner<real,normalize>::n;
   const int N = dminner<real,normalize>::ws.size();
   const int tau = N*n;
   assert(N > 0);
   // Copy channel for access within Q()
   dminner<real,normalize>::mychan = dynamic_cast<const bsid&>(chan);
   // Set channel block size to q-ary symbol size
   dminner<real,normalize>::mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const double Pd = dminner<real,normalize>::mychan.get_pd();
   const int I = bsid::compute_I(tau, Pd);
   const int xmax = bsid::compute_xmax(tau, Pd, I);
   const int dxmax = bsid::compute_xmax(n, Pd);
   dminner<real,normalize>::checkforchanges(I, xmax);
   // Initialize & perform forward-backward algorithm
   fba2<real,bool,normalize>::init(N, n, q, I, xmax, dxmax, dminner<real,normalize>::th_inner, dminner<real,normalize>::th_outer);
   fba2<real,bool,normalize>::prepare(rx);
   // Compute and normalize results
   libbase::matrix<real> p;
   fba2<real,bool,normalize>::work_results(rx,p);
   dminner<real,normalize>::normalize_results(p,ptable);
   }

// description output

template <class real, bool normalize>
std::string dminner2<real,normalize>::description() const
   {
   std::ostringstream sout;
   sout << "Alternative " << dminner<real,normalize>::description();
   return sout.str();
   }

// object serialization - saving

template <class real, bool normalize>
std::ostream& dminner2<real,normalize>::serialize(std::ostream& sout) const
   {
   return dminner<real,normalize>::serialize(sout);
   }

// object serialization - loading

template <class real, bool normalize>
std::istream& dminner2<real,normalize>::serialize(std::istream& sin)
   {
   return dminner<real,normalize>::serialize(sin);
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2<logrealfast,false>;
template <>
const serializer dminner2<logrealfast,false>::shelper \
   = serializer("modulator", "dminner2<logrealfast>", dminner2<logrealfast,false>::create);

template class dminner2<double,true>;
template <>
const serializer dminner2<double,true>::shelper \
   = serializer("modulator", "dminner2<double>", dminner2<double,true>::create);

}; // end namespace
