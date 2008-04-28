#ifndef __serializer_libcomm_h
#define __serializer_libcomm_h


// Utilities
#include "gf.h"

// Arithmetic Types
#include "mpgnu.h"
#include "mpreal.h"
#include "logreal.h"
#include "logrealfast.h"


// Channels
#include "channel.h"
#include "awgn.h"
#include "laplacian.h"
#include "lapgauss.h"
#include "bsid.h"
#include "bsc.h"
#include "qsc.h"

// Modulators
#include "modulator.h"
#include "mpsk.h"
#include "qam.h"
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

// Signal Mappers
#include "mapper.h"
#include "map_straight.h"
#include "map_interleaved.h"

// Puncture Patterns
#include "puncture.h"
#include "puncture_file.h"
#include "puncture_null.h"
#include "puncture_stipple.h"


// Experiments
#include "commsys.h"
#include "commsys_prof_burst.h"
#include "commsys_prof_pos.h"
#include "commsys_prof_sym.h"
#include "commsys_hist_symerr.h"


namespace libcomm {

/*!
   \brief   Communications Library Serializer.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 2.00 (13 Oct 2006)
   - added version object to make this class accessible.

   \version 2.10 (6 Nov 2006)
   - defined class and associated data within "libcomm" namespace.

   \version 2.20 (1 Nov 2007)
   - added bsid and watermarkcode.

   \version 2.21 (7 Nov 2007)
   - resolved ambiguity with bsid and mpsk direct bases, by removing the
    direct base.

   \version 2.22 (13-14 Dec 2007)
   - added grscc<> variants for GF(2), GF(2^4)
   - added gnrcc<> variants for GF(2), GF(2^4)

   \version 2.23 (3 Jan 2008)
   - added qam

   \version 2.24 (21 Jan 2008)
   - Added bsid again (since this is no longer a base class of anything)
   - Added lapgauss channel

   \version 2.25 (24 Jan 2008)
   - Added commsys experiment type

   \version 2.26 (25 Jan 2008)
   - Modified commsys to commsys<sigspace>
   - Added BSC channel

   \version 2.27 (28 Jan 2008)
   - Added modulator<bool>
   - Replace modulator<bool> by mpsk, due to change in watermarkcode inheritance

   \version 2.28 (12 Feb 2008)
   - Added GF(q) commsys classes
   - Added q-ary symmetric channel

   \version 2.29 (19 Feb 2008)
   - Added symbol-error histogram and position profiler for commsys<bool>
   - Added symbol-value profiler for commsys<bool>

   \version 2.30 (15 Apr 2008)
   - Added burst-error profiler for commsys<bool>

   \version 2.31 (25-28 Apr 2008)
   - Added straight and interleaved signal mappers
*/

// Serialization support
class serializer_libcomm : private
   qsc< libbase::gf<1,0x3> >, qsc< libbase::gf<2,0x7> >, qsc< libbase::gf<3,0xB> >, qsc< libbase::gf<4,0x13> >,
   awgn, laplacian, lapgauss, bsid, bsc,
   mpsk, qam, watermarkcode<libbase::logrealfast>,
   nrcc, rscc, dvbcrsc,
   grscc< libbase::gf<1,0x3> >, grscc< libbase::gf<2,0x7> >, grscc< libbase::gf<3,0xB> >, grscc< libbase::gf<4,0x13> >,
   gnrcc< libbase::gf<1,0x3> >, gnrcc< libbase::gf<2,0x7> >, gnrcc< libbase::gf<3,0xB> >, gnrcc< libbase::gf<4,0x13> >,
   onetimepad, padded, berrou, flat, helical, rand_lut, rectangular, shift_lut, uniform_lut, named_lut,
   uncoded, mapcc<libbase::logrealfast>, turbo<libbase::logrealfast,libbase::logrealfast>, diffturbo<libbase::logrealfast>,
   map_interleaved,
   puncture_file, puncture_null, puncture_stipple,
   commsys< libbase::gf<1,0x3> >, commsys< libbase::gf<2,0x7> >, commsys< libbase::gf<3,0xB> >, commsys< libbase::gf<4,0x13> >,
   commsys<sigspace>, commsys<bool>,
   commsys<bool,commsys_prof_burst>,
   commsys<bool,commsys_prof_pos>,
   commsys<bool,commsys_prof_sym>,
   commsys<bool,commsys_hist_symerr>
{
public:
   serializer_libcomm() {};
};

}; // end namespace

#endif
