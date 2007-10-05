#include "watermarkcode.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real> void watermarkcode<real>::init()
   {
   }

template <class real> void watermarkcode<real>::free()
   {
   }

// constructor / destructor

template <class real> watermarkcode<real>::watermarkcode()
   {
   }

template <class real> watermarkcode<real>::watermarkcode(const int tau)
   {
   watermarkcode::tau = tau;
   init();
   }

// encoding and decoding functions

template <class real> void watermarkcode<real>::encode(libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   }

template <class real> void watermarkcode<real>::translate(const libbase::matrix<double>& ptable)
   {
   }

template <class real> void watermarkcode<real>::decode(libbase::vector<int>& decoded)
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
   sout << tau << "\n";
   return sout;
   }

// object serialization - loading

template <class real> std::istream& watermarkcode<real>::serialize(std::istream& sin)
   {
   free();
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
