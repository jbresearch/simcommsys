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
#include "blockmodem.h"
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

// Signal Mappers
#include "mapper.h"
#include "map_straight.h"
#include "map_interleaved.h"
#include "map_stipple.h"

// Experiments
#include "commsys.h"
#include "commsys_simulator.h"
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
   uncoded, mapcc<libbase::logrealfast>, turbo<libbase::logrealfast,libbase::logrealfast>,
   onetimepad<double>, padded<double>, berrou<double>, flat<double>, helical<double>, rand_lut<double>, rectangular<double>, shift_lut<double>, uniform_lut<double>, named_lut<double>,
   onetimepad<libbase::logrealfast>, padded<libbase::logrealfast>, berrou<libbase::logrealfast>, flat<libbase::logrealfast>, helical<libbase::logrealfast>, rand_lut<libbase::logrealfast>, rectangular<libbase::logrealfast>, shift_lut<libbase::logrealfast>, uniform_lut<libbase::logrealfast>, named_lut<libbase::logrealfast>,
   map_interleaved, map_stipple
{
private:
   typedef libbase::logrealfast  logrealfast;
private:
   // Interleavers
   //onetimepad<double>	_onetimepad_double;
   //padded<double>	_padded_double;
   //berrou<double>	_berrou_double;
   //flat<double>   	_flat_double;
   //helical<double>	_helical_double;
   //rand_lut<double>	_rand_lut_double;
   //rectangular<double>	_rectangular_double;
   //shift_lut<double>	_shift_lut_double;
   //uniform_lut<double>	_uniform_lut_double;
   //named_lut<double>	_named_lut_double;
   //onetimepad<logrealfast>	_onetimepad_double;
   //padded<logrealfast>	_padded_double;
   //berrou<logrealfast>	_berrou_double;
   //flat<logrealfast>   	_flat_double;
   //helical<logrealfast>	_helical_double;
   //rand_lut<logrealfast>	_rand_lut_double;
   //rectangular<logrealfast>	_rectangular_double;
   //shift_lut<logrealfast>	_shift_lut_double;
   //uniform_lut<logrealfast>	_uniform_lut_double;
   //named_lut<logrealfast>	_named_lut_double;
   // Modulators
   dminner<logrealfast,false>       _dminner_logrealfast;
   dminner2<logrealfast,false>      _dminner2_logrealfast;
   // Experiments
   commsys< libbase::gf<1,0x3> >       _commsys_gf1;
   commsys< libbase::gf<2,0x7> >       _commsys_gf2;
   commsys< libbase::gf<3,0xB> >       _commsys_gf3;
   commsys< libbase::gf<4,0x13> >      _commsys_gf4;
   commsys<sigspace>                   _commsys_sigspace;
   commsys<bool>                       _commsys_bool;
   // Experiments
   commsys_simulator< libbase::gf<1,0x3> >       _commsys_simulator_gf1;
   commsys_simulator< libbase::gf<2,0x7> >       _commsys_simulator_gf2;
   commsys_simulator< libbase::gf<3,0xB> >       _commsys_simulator_gf3;
   commsys_simulator< libbase::gf<4,0x13> >      _commsys_simulator_gf4;
   commsys_simulator<sigspace>                   _commsys_simulator_sigspace;
   commsys_simulator<bool>                       _commsys_simulator_bool;
   commsys_simulator<bool,commsys_prof_burst>    _commsys_simulator_bool_prof_burst;
   commsys_simulator<bool,commsys_prof_pos>      _commsys_simulator_bool_prof_pos;
   commsys_simulator<bool,commsys_prof_sym>      _commsys_simulator_bool_prof_sym;
   commsys_simulator<bool,commsys_hist_symerr>   _commsys_simulator_bool_hist_symerr;
   commsys_threshold<bool>             _commsys_threshold_bool;
public:
   serializer_libcomm() {};
};

}; // end namespace

#endif
