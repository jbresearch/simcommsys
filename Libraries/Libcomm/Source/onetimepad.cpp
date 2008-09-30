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

const libbase::serializer onetimepad::shelper("interleaver", "onetimepad", onetimepad::create);

// construction and destruction

onetimepad::onetimepad()
   {
   encoder = NULL;
   }

onetimepad::onetimepad(const fsm& encoder, const int tau, const bool terminated, const bool renewable)
   {
   onetimepad::terminated = terminated;
   onetimepad::renewable = renewable;
   onetimepad::encoder = encoder.clone();
   onetimepad::m = encoder.mem_order();
   onetimepad::K = encoder.num_inputs();
   pad.init(tau);
   libbase::trace << "DEBUG (onetimepad): constructed interleaver (tau=" << tau << ", m=" << m << ", K=" << K << ")\n";
   }

onetimepad::onetimepad(const onetimepad& x)
   {
   terminated = x.terminated;
   renewable = x.renewable;
   encoder = x.encoder->clone();
   m = x.m;
   K = x.K;
   pad = x.pad;
   r = x.r;
   }

onetimepad::~onetimepad()
   {
   if(encoder != NULL)
      delete encoder;
   }

// inter-frame operations

void onetimepad::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   advance();
   }

void onetimepad::advance()
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

void onetimepad::transform(const vector<int>& in, vector<int>& out) const
   {
   const int tau = pad.size();
   assertalways(in.size() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      out(t) = (in(t) + pad(t)) % K;
   }

void onetimepad::transform(const matrix<double>& in, matrix<double>& out) const
   {
   const int tau = pad.size();
   assertalways(in.ysize() == K);
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      for(int i=0; i<K; i++)
         out(t, i) = in(t, (i+pad(t))%K);
   }

void onetimepad::inverse(const matrix<double>& in, matrix<double>& out) const
   {
   const int tau = pad.size();
   assertalways(in.ysize() == K);
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      for(int i=0; i<K; i++)
         out(t, (i+pad(t))%K) = in(t, i);
   }

void onetimepad::transform(const matrix<libbase::logrealfast>& in, matrix<libbase::logrealfast>& out) const
   {
   const int tau = pad.size();
   assertalways(in.ysize() == K);
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      for(int i=0; i<K; i++)
         out(t, i) = in(t, (i+pad(t))%K);
   }

void onetimepad::inverse(const matrix<libbase::logrealfast>& in, matrix<libbase::logrealfast>& out) const
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

std::string onetimepad::description() const
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

std::ostream& onetimepad::serialize(std::ostream& sout) const
   {
   sout << int(terminated) << "\n";
   sout << int(renewable) << "\n";
   sout << pad.size() << "\n";
   sout << encoder;
   return sout;
   }

// object serialization - loading

std::istream& onetimepad::serialize(std::istream& sin)
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

}; // end namespace
