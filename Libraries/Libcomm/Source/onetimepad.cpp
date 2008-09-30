/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "onetimepad.h"
#include <sstream>

namespace libcomm {

using std::cerr;

using libbase::vector;
using libbase::matrix;

// construction and destruction

template <class real>
onetimepad<real>::onetimepad()
   {
   encoder = NULL;
   }

template <class real>
onetimepad<real>::onetimepad(const fsm& encoder, const int tau, const bool terminated, const bool renewable)
   {
   onetimepad<real>::terminated = terminated;
   onetimepad<real>::renewable = renewable;
   onetimepad<real>::encoder = encoder.clone();
   onetimepad<real>::m = encoder.mem_order();
   onetimepad<real>::K = encoder.num_inputs();
   pad.init(tau);
   libbase::trace << "DEBUG (onetimepad): constructed interleaver (tau=" << tau << ", m=" << m << ", K=" << K << ")\n";
   }

template <class real>
onetimepad<real>::onetimepad(const onetimepad& x)
   {
   terminated = x.terminated;
   renewable = x.renewable;
   encoder = x.encoder->clone();
   m = x.m;
   K = x.K;
   pad = x.pad;
   r = x.r;
   }

template <class real>
onetimepad<real>::~onetimepad()
   {
   if(encoder != NULL)
      delete encoder;
   }

// inter-frame operations

template <class real>
void onetimepad<real>::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   advance();
   }

template <class real>
void onetimepad<real>::advance()
   {
   static bool initialised = false;

   // do not advance if this interleaver is not renewable
   if(!renewable && initialised)
      return;

   const int tau = pad.size();
   // fill in pad
   if(terminated)
      {
      int t;
      for(t=0; t<tau-m; t++)
         pad(t) = r.ival(K);
      for(t=tau-m; t<tau; t++)
         pad(t) = fsm::tail;
      // run through the encoder once, so that we work out the tail bits
      encoder->reset(0);
      for(t=0; t<tau; t++)
         encoder->step(pad(t));
      }
   else
      {
      for(int t=0; t<tau; t++)
         pad(t) = r.ival(K);
      }

   initialised = true;
   }

// transform functions

template <class real>
void onetimepad<real>::transform(const vector<int>& in, vector<int>& out) const
   {
   const int tau = pad.size();
   assertalways(in.size() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      out(t) = (in(t) + pad(t)) % K;
   }

template <class real>
void onetimepad<real>::transform(const matrix<real>& in, matrix<real>& out) const
   {
   const int tau = pad.size();
   assertalways(in.ysize() == K);
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      for(int i=0; i<K; i++)
         out(t, i) = in(t, (i+pad(t))%K);
   }

template <class real>
void onetimepad<real>::inverse(const matrix<real>& in, matrix<real>& out) const
   {
   const int tau = pad.size();
   assertalways(in.ysize() == K);
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      for(int i=0; i<K; i++)
         out(t, (i+pad(t))%K) = in(t, i);
   }

// description output

template <class real>
std::string onetimepad<real>::description() const
   {
   std::ostringstream sout;
   sout << "One-Time-Pad Interleaver (";
   if(terminated)
      sout << "terminated";
   if(terminated && renewable)
      sout << ", ";
   if(renewable)
      sout << "renewable";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& onetimepad<real>::serialize(std::ostream& sout) const
   {
   sout << int(terminated) << "\n";
   sout << int(renewable) << "\n";
   sout << pad.size() << "\n";
   sout << encoder;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& onetimepad<real>::serialize(std::istream& sin)
   {
   int temp;
   sin >> temp;
   terminated = temp != 0;
   sin >> temp;
   renewable = temp != 0;
   sin >> temp;
   pad.init(temp);
   sin >> encoder;
   m = encoder->mem_order();
   K = encoder->num_inputs();
   return sin;
   }

// Explicit instantiations

template class onetimepad<double>;
template <>
const libbase::serializer onetimepad<double>::shelper("interleaver", "onetimepad<double>", onetimepad<double>::create);

template class onetimepad<libbase::logrealfast>;
template <>
const libbase::serializer onetimepad<libbase::logrealfast>::shelper("interleaver", "onetimepad<logrealfast>", onetimepad<libbase::logrealfast>::create);

}; // end namespace
