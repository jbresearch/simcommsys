/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mapcc.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real> void mapcc<real>::init()
   {
   bcjr<real>::init(*encoder, tau, !circular, endatzero, circular);

   m = endatzero ? encoder->mem_order() : 0;
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

template <class real> mapcc<real>::mapcc(const fsm& encoder, const int tau, const bool endatzero, const bool circular)
   {
   mapcc::encoder = encoder.clone();
   mapcc::tau = tau;
   mapcc::endatzero = endatzero;
   mapcc::circular = circular;
   init();
   }

// encoding and decoding functions

template <class real> void mapcc<real>::encode(libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   // Initialise result vector
   encoded.init(tau);
   // Reset the encoder to zero state
   encoder->reset(0);
   // When dealing with a circular system, perform first pass to determine end state,
   // then reset to the corresponding circular state.
   if(circular)
      {
      for(int t=0; t<tau; t++)
         encoder->advance(source(t));
      encoder->resetcircular();
      }
   // Encode source stream
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
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << "MAP-decoded Convolutional Code (" << output_bits() << "," << input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class real> std::ostream& mapcc<real>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   sout << int(endatzero) << "\n";
   sout << int(circular) << "\n";
   return sout;
   }

// object serialization - loading

template <class real> std::istream& mapcc<real>::serialize(std::istream& sin)
   {
   int temp;
   free();
   sin >> encoder;
   sin >> tau;
   sin >> temp;
   endatzero = temp != 0;
   sin >> temp;
   circular = temp != 0;
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

template class mapcc<mpreal>;
template <> const serializer mapcc<mpreal>::shelper = serializer("codec", "mapcc<mpreal>", mapcc<mpreal>::create);

template class mapcc<mpgnu>;
template <> const serializer mapcc<mpgnu>::shelper = serializer("codec", "mapcc<mpgnu>", mapcc<mpgnu>::create);

template class mapcc<logreal>;
template <> const serializer mapcc<logreal>::shelper = serializer("codec", "mapcc<logreal>", mapcc<logreal>::create);

template class mapcc<logrealfast>;
template <> const serializer mapcc<logrealfast>::shelper = serializer("codec", "mapcc<logrealfast>", mapcc<logrealfast>::create);

}; // end namespace
