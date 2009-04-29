/*!
   \file

   \section svn Version Control
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
real dminner2<real,normalize>::R(int d, int i, const array1b_t& r) const
   {
   const int n = Base::n;
   // 'tx' is the vector of transmitted symbols that we're considering
   array1b_t tx;
   tx.init(n);
   const int w = Base::ws(i);
   const int s = Base::lut(d);
   // NOTE: we transmit the low-order bits first
   for(int bit=0, t=s^w; bit<n; bit++, t >>= 1)
      tx(bit) = (t&1);
   // compute the conditional probability
   return Base::mychan.receive(tx, r);
   }

// Setup procedure

template <class real, bool normalize>
void dminner2<real,normalize>::init(const channel<bool>& chan)
   {
   // Inherit block size from last modulation step
   const int q = 1<<Base::k;
   const int n = Base::n;
   const int N = Base::ws.size();
   const int tau = N*n;
   assert(N > 0);
   // Copy channel for access within R()
   Base::mychan = dynamic_cast<const bsid&>(chan);
   // Set channel block size to q-ary symbol size
   Base::mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const double Pd = Base::mychan.get_pd();
   const int I = bsid::compute_I(tau, Pd);
   const int xmax = bsid::compute_xmax(tau, Pd, I);
   const int dxmax = bsid::compute_xmax(n, Pd);
   Base::checkforchanges(I, xmax);
   // Initialize forward-backward algorithm
   FBA::init(N, n, q, I, xmax, dxmax, Base::th_inner, Base::th_outer);
   }

// encoding and decoding functions

template <class real, bool normalize>
void dminner2<real,normalize>::dodemodulate(const channel<bool>& chan, const array1b_t& rx, array1vd_t& ptable)
   {
   init(chan);
   array1vr_t p;
   FBA::decode(rx,p);
   Base::normalize_results(p,ptable);
   }

template <class real, bool normalize>
void dminner2<real,normalize>::dodemodulate(const channel<bool>& chan, const array1b_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   init(chan);
   array1vr_t p;
   FBA::decode(rx,app,p);
   Base::normalize_results(p,ptable);
   }

// description output

template <class real, bool normalize>
std::string dminner2<real,normalize>::description() const
   {
   std::ostringstream sout;
   sout << "Symbol-level " << Base::description();
   return sout.str();
   }

// object serialization - saving

template <class real, bool normalize>
std::ostream& dminner2<real,normalize>::serialize(std::ostream& sout) const
   {
   return Base::serialize(sout);
   }

// object serialization - loading

template <class real, bool normalize>
std::istream& dminner2<real,normalize>::serialize(std::istream& sin)
   {
   return Base::serialize(sin);
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2<logrealfast,false>;
template <>
const serializer dminner2<logrealfast,false>::shelper
   = serializer("blockmodem", "dminner2<logrealfast>", dminner2<logrealfast,false>::create);

template class dminner2<double,true>;
template <>
const serializer dminner2<double,true>::shelper
   = serializer("blockmodem", "dminner2<double>", dminner2<double,true>::create);

}; // end namespace
