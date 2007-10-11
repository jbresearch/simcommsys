#include "watermarkcode.h"
#include <sstream>

namespace libcomm {

// LUT creation

template <class real> int watermarkcode<real>::fill(int i, libbase::bitfield suffix, int w)
   {
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
   
// initialization / de-allocation

template <class real> void watermarkcode<real>::init()
   {
   using libbase::weight;
   lut.init(num_inputs());
   fill(0,"",n);
   r.seed(s);
#ifndef NDEBUG
   // Display LUT when debugging
   using libbase::trace;
   trace << "LUT (k=" << k << ", n=" << n << "):\n";
   libbase::bitfield b;
   b.resize(n);
   for(int i=0; i<lut.size(); i++)
      {
      b = lut(i);
      trace << i << "\t" << b << "\t" << weight(b) << "\n";
      }
#endif
   }

template <class real> void watermarkcode<real>::free()
   {
   }

// constructor / destructor

template <class real> watermarkcode<real>::watermarkcode()
   {
   }

template <class real> watermarkcode<real>::watermarkcode(const int N, const int n, const int k, const int s)
   {
   watermarkcode::N = N;
   watermarkcode::n = n;
   watermarkcode::k = k;
   watermarkcode::s = s;
   init();
   }

// implementations of channel-specific metrics for fba

template <class real> double watermarkcode<real>::P(const int a, const int b)
   {
   return 0;
   }
   
template <class real> double watermarkcode<real>::Q(const int a, const int b, const int i, const int s)
   {
   return 0;
   }
   
// encoding and decoding functions

template <class real> void watermarkcode<real>::encode(libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   // Initialise result vector
   encoded.init(N);
   // Encode source stream
   assert(source.size() == N);
   for(int i=0; i<N; i++)
      encoded(i) = lut(source(i)) ^ r.ival(num_outputs());
   }

template <class real> void watermarkcode<real>::translate(const libbase::matrix<double>& ptable)
   {
   }

template <class real> void watermarkcode<real>::decode(libbase::vector<int>& decoded)
   {
   }

template <class real> void watermarkcode<real>::decode(libbase::matrix<double>& decoded)
   {
   }

// description output

template <class real> std::string watermarkcode<real>::description() const
   {
   std::ostringstream sout;
   sout << "Watermark Code (" << output_bits() << "," << input_bits() << "), ";
   return sout.str();
   }

// object serialization - saving

template <class real> std::ostream& watermarkcode<real>::serialize(std::ostream& sout) const
   {
   sout << N << "\n";
   sout << n << "\n";
   sout << k << "\n";
   sout << s << "\n";
   return sout;
   }

// object serialization - loading

template <class real> std::istream& watermarkcode<real>::serialize(std::istream& sin)
   {
   free();
   sin >> N;
   sin >> n;
   sin >> k;
   sin >> s;
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

#define VERSION 1.00

template class watermarkcode<mpreal>;
template <> const serializer watermarkcode<mpreal>::shelper = serializer("codec", "watermarkcode<mpreal>", watermarkcode<mpreal>::create);
template <> const vcs watermarkcode<mpreal>::version = vcs("Watermark Codec module (watermarkcode<mpreal>)", VERSION);

template class watermarkcode<mpgnu>;
template <> const serializer watermarkcode<mpgnu>::shelper = serializer("codec", "watermarkcode<mpgnu>", watermarkcode<mpgnu>::create);
template <> const vcs watermarkcode<mpgnu>::version = vcs("Watermark Codec module (watermarkcode<mpgnu>)", VERSION);

template class watermarkcode<logreal>;
template <> const serializer watermarkcode<logreal>::shelper = serializer("codec", "watermarkcode<logreal>", watermarkcode<logreal>::create);
template <> const vcs watermarkcode<logreal>::version = vcs("Watermark Codec module (watermarkcode<logreal>)", VERSION);

template class watermarkcode<logrealfast>;
template <> const serializer watermarkcode<logrealfast>::shelper = serializer("codec", "watermarkcode<logrealfast>", watermarkcode<logrealfast>::create);
template <> const vcs watermarkcode<logrealfast>::version = vcs("Watermark Codec module (watermarkcode<logrealfast>)", VERSION);

}; // end namespace
