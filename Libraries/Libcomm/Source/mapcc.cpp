#include "mapcc.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real> void mapcc<real>::init()
   {
   bcjr<real>::init(*encoder, tau, true, true);

   m = encoder->mem_order();
   M = encoder->num_states();
   K = encoder->num_inputs();
   N = encoder->num_outputs();
   }

template <class real> void mapcc<real>::free()
   {
   if(encoder != NULL)
      delete encoder;
   }

// constructor / destructor

template <class real> mapcc<real>::mapcc()
   {
   encoder = NULL;
   }

template <class real> mapcc<real>::mapcc(const fsm& encoder, const int tau)
   {
   mapcc::encoder = encoder.clone();
   mapcc::tau = tau;
   init();
   }

// encoding and decoding functions

template <class real> void mapcc<real>::encode(libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   // Initialise result vector
   encoded.init(tau);
   // Encode source stream
   encoder->reset(0);
   for(int t=0; t<tau; t++)
      encoded(t) = encoder->step(source(t));
   }

template <class real> void mapcc<real>::translate(const libbase::matrix<double>& ptable)
   {
   using std::cerr;
   // Compute factors / sizes & check validity
   const int S = ptable.ysize();
   const int s = int(libbase::round(log(double(N))/log(double(S))));
   if(N != pow(double(S), s))
      {
      cerr << "FATAL ERROR (mapcc): each encoder output (" << N << ") must be";
      cerr << " represented by an integral number of modulation symbols (" << S << ").";
      cerr << " Suggested number of mod. symbols/encoder output was " << s << ".\n";
      exit(1);
      }
   if(ptable.xsize() != tau*s)
      {
      cerr << "FATAL ERROR (mapcc): demodulation table should have " << tau*s;
      cerr << " symbols, not " << ptable.xsize() << ".\n";
      exit(1);
      }
   // Initialize results vector
   R.init(tau, N);
   // Compute the Input statistics for the BCJR Algorithm
   for(int t=0; t<tau; t++)
      for(int x=0; x<N; x++)
         {
         R(t, x) = 1;
         for(int i=0, thisx = x; i<s; i++, thisx /= S)
            R(t, x) *= ptable(t*s+i, thisx % S);
         }
   }

template <class real> void mapcc<real>::decode(libbase::vector<int>& decoded)
   {
   // Initialize results vectors
   ri.init(tau, K);
   ro.init(tau, N);
   decoded.init(tau);
   // Decode using BCJR algorithm
   bcjr<real>::decode(R, ri, ro);
   // Decide which input sequence was most probable, based on BCJR stats.
   for(int t=0; t<tau; t++)
      {
      decoded(t) = 0;
      for(int i=1; i<K; i++)
         if(ri(t, i) > ri(t, decoded(t)))
            decoded(t) = i;
      }
   }

// description output

template <class real> std::string mapcc<real>::description() const
   {
   std::ostringstream sout;
   sout << "Terminated Convolutional Code (" << output_bits() << "," << input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class real> std::ostream& mapcc<real>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   return sout;
   }

// object serialization - loading

template <class real> std::istream& mapcc<real>::serialize(std::istream& sin)
   {
   free();
   sin >> encoder;
   sin >> tau;
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

using libbase::serializer;
using libbase::vcs;

#define VERSION 1.52

template class mapcc<mpreal>;
template <> const serializer mapcc<mpreal>::shelper = serializer("codec", "mapcc<mpreal>", mapcc<mpreal>::create);
template <> const vcs mapcc<mpreal>::version = vcs("Maximum A-Posteriori Decoder module (mapcc<mpreal>)", VERSION);

template class mapcc<mpgnu>;
template <> const serializer mapcc<mpgnu>::shelper = serializer("codec", "mapcc<mpgnu>", mapcc<mpgnu>::create);
template <> const vcs mapcc<mpgnu>::version = vcs("Maximum A-Posteriori Decoder module (mapcc<mpgnu>)", VERSION);

template class mapcc<logreal>;
template <> const serializer mapcc<logreal>::shelper = serializer("codec", "mapcc<logreal>", mapcc<logreal>::create);
template <> const vcs mapcc<logreal>::version = vcs("Maximum A-Posteriori Decoder module (mapcc<logreal>)", VERSION);

template class mapcc<logrealfast>;
template <> const serializer mapcc<logrealfast>::shelper = serializer("codec", "mapcc<logrealfast>", mapcc<logrealfast>::create);
template <> const vcs mapcc<logrealfast>::version = vcs("Maximum A-Posteriori Decoder module (mapcc<logrealfast>)", VERSION);

}; // end namespace
