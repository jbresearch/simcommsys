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
#include "dminner.h"
#include "dminner2.h"

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
#include "commsys_threshold.h"


namespace libcomm {

/*!
   \brief   Communications Library Serializer.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

// Serialization support
class serializer_libcomm : private
   qsc< libbase::gf<1,0x3> >, qsc< libbase::gf<2,0x7> >, qsc< libbase::gf<3,0xB> >, qsc< libbase::gf<4,0x13> >,
   awgn, laplacian, lapgauss, bsid, bsc,
   mpsk, qam,
   nrcc, rscc, dvbcrsc,
   grscc< libbase::gf<1,0x3> >, grscc< libbase::gf<2,0x7> >, grscc< libbase::gf<3,0xB> >, grscc< libbase::gf<4,0x13> >,
   gnrcc< libbase::gf<1,0x3> >, gnrcc< libbase::gf<2,0x7> >, gnrcc< libbase::gf<3,0xB> >, gnrcc< libbase::gf<4,0x13> >,
   onetimepad, padded, berrou, flat, helical, rand_lut, rectangular, shift_lut, uniform_lut, named_lut,
   uncoded, mapcc<libbase::logrealfast>, turbo<libbase::logrealfast,libbase::logrealfast>, diffturbo<libbase::logrealfast>,
   map_interleaved,
   puncture_file, puncture_null, puncture_stipple
{
private:
   // Modulators
   dminner<libbase::logrealfast,false>       _dminner_logrealfast;
   dminner2<libbase::logrealfast,false>      _dminner2_logrealfast;
   // Experiments
   commsys< libbase::gf<1,0x3> >       _commsys_gf1;
   commsys< libbase::gf<2,0x7> >       _commsys_gf2;
   commsys< libbase::gf<3,0xB> >       _commsys_gf3;
   commsys< libbase::gf<4,0x13> >      _commsys_gf4;
   commsys<sigspace>                   _commsys_sigspace;
   commsys<bool>                       _commsys_bool;
   commsys<bool,commsys_prof_burst>    _commsys_bool_prof_burst;
   commsys<bool,commsys_prof_pos>      _commsys_bool_prof_pos;
   commsys<bool,commsys_prof_sym>      _commsys_bool_prof_sym;
   commsys<bool,commsys_hist_symerr>   _commsys_bool_hist_symerr;
   commsys_threshold<bool>             _commsys_threshold_bool;
public:
   serializer_libcomm() {};
};

}; // end namespace

#endif
