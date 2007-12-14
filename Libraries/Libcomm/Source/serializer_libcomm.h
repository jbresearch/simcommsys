#ifndef __serializer_libcomm_h
#define __serializer_libcomm_h

#include "vcs.h"

// Utilities
#include "gf.h"

// Channels
#include "channel.h"
#include "awgn.h"
#include "laplacian.h"
#include "bsid.h"

// Modulators
#include "modulator.h"
#include "mpsk.h"
#include "watermarkcode.h"

// Convolutional Encoders
#include "fsm.h"
#include "nrcc.h"
#include "rscc.h"
#include "dvbcrsc.h"
#include "grscc.h"
#include "gnrcc.h"

// Interleavers
#include "interleaver.h"
#include "onetimepad.h"
#include "padded.h"
// LUT Interleavers
#include "lut_interleaver.h"
#include "berrou.h"
#include "flat.h"
#include "helical.h"
#include "rand_lut.h"
#include "rectangular.h"
#include "shift_lut.h"
#include "uniform_lut.h"
// Named LUT
#include "named_lut.h"
#include "file_lut.h"
#include "stream_lut.h"
#include "vale96int.h"

// Codecs
#include "codec.h"
#include "uncoded.h"
#include "mapcc.h"
#include "turbo.h"
#include "diffturbo.h"

// Puncture Patterns
#include "puncture.h"
#include "puncture_file.h"
#include "puncture_null.h"
#include "puncture_stipple.h"


// Arithmetic Types
#include "mpgnu.h"
#include "mpreal.h"
#include "logreal.h"
#include "logrealfast.h"

/*
  Version 2.00 (13 Oct 2006)
  * added version object to make this class accessible.

  Version 2.10 (6 Nov 2006)
  * defined class and associated data within "libcomm" namespace.

  Version 2.20 (1 Nov 2007)
  * added bsid and watermarkcode.

  Version 2.21 (7 Nov 2007)
  * resolved ambiguity with bsid and mpsk direct bases, by removing the
    direct base.

  Version 2.22 (13-14 Dec 2007)
  * added grscc<> variants for GF(2^4)
  * added gnrcc<> variants for GF(2^4)
*/

namespace libcomm {

// Serialization support
class serializer_libcomm : private
   awgn, laplacian,
   watermarkcode<libbase::logrealfast>,
   nrcc, rscc, dvbcrsc,
   grscc< libbase::gf<4,0x13> >,
   gnrcc< libbase::gf<4,0x13> >,
   onetimepad, padded, berrou, flat, helical, rand_lut, rectangular, shift_lut, uniform_lut, named_lut,
   uncoded, mapcc<libbase::logrealfast>, turbo<libbase::logrealfast,libbase::logrealfast>, diffturbo<libbase::logrealfast>,
   puncture_file, puncture_null, puncture_stipple
{
   static const libbase::vcs version;
public:
   serializer_libcomm() {};
};

}; // end namespace

#endif
