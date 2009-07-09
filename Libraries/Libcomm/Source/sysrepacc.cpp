/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "sysrepacc.h"
#include <sstream>
#include <iomanip>

namespace libcomm {

// encoding and decoding functions

template <class real, class dbl>
void sysrepacc<real, dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   array1i_t parity;
   Base::encode(source, parity);
   encoded.init(This::output_block_size());
   encoded.segment(0, source.size()).copyfrom(source);
   encoded.segment(source.size(), parity.size()).copyfrom(parity);
   }

template <class real, class dbl>
void sysrepacc<real, dbl>::init_decoder(const array1vd_t& ptable)
   {
   // Inherit sizes
   const int Ns = Base::input_block_size();
   const int Np = Base::output_block_size();
   const int q = Base::num_inputs();
   const int qo = Base::num_outputs();
   assertalways(q == qo);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Divide ptable for input and output sides
   const array1vd_t iptable = ptable.extract(0, Ns);
   const array1vd_t optable = ptable.extract(Ns, Np);
   // Perform standard decoder initialization
   Base::init_decoder(optable);
   // Determine intrinsic source statistics (natural)
   // from the channel
   for (int i = 0; i < Ns; i++)
      for (int x = 0; x < q; x++) // 'x' is the input symbol
         rp(i)(x) *= dbl(iptable(i)(x));
   //    BCJR::normalize(rp);
   }

template <class real, class dbl>
void sysrepacc<real, dbl>::init_decoder(const array1vd_t& ptable,
      const array1vd_t& app)
   {
   // Inherit sizes
   const int Ns = Base::input_block_size();
   const int Np = Base::output_block_size();
   const int q = Base::num_inputs();
   const int qo = Base::num_outputs();
   assertalways(q == qo);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Divide ptable for input and output sides
   const array1vd_t iptable = ptable.extract(0, Ns);
   const array1vd_t optable = ptable.extract(Ns, Np);
   // Perform standard decoder initialization
   Base::init_decoder(optable, app);
   // Determine intrinsic source statistics (natural)
   // from the channel
   for (int i = 0; i < Ns; i++)
      for (int x = 0; x < q; x++) // 'x' is the input symbol
         rp(i)(x) *= iptable(i)(x);
   //    BCJR::normalize(rp);
   }

// description output

template <class real, class dbl>
std::string sysrepacc<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Systematic " << Base::description();
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& sysrepacc<real, dbl>::serialize(std::ostream& sout) const
   {
   return Base::serialize(sout);
   }

// object serialization - loading

template <class real, class dbl>
std::istream& sysrepacc<real, dbl>::serialize(std::istream& sin)
   {
   return Base::serialize(sin);
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;
using libbase::serializer;

template class sysrepacc<float, float>
template <>
const serializer sysrepacc<float, float>::shelper = serializer("codec",
      "sysrepacc<float>", sysrepacc<float, float>::create);

template class sysrepacc<double>
template <>
const serializer sysrepacc<double>::shelper = serializer("codec",
      "sysrepacc<double>", sysrepacc<double>::create);

template class sysrepacc<logrealfast>
template <>
const serializer sysrepacc<logrealfast>::shelper = serializer("codec",
      "sysrepacc<logrealfast>", sysrepacc<logrealfast>::create);

template class sysrepacc<logrealfast, logrealfast>
template <>
const serializer sysrepacc<logrealfast, logrealfast>::shelper = serializer(
      "codec", "sysrepacc<logrealfast,logrealfast>", sysrepacc<logrealfast,
            logrealfast>::create);

} // end namespace
