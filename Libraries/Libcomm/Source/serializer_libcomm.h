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
#include "bsid2d.h"
#include "bsc.h"
#include "qsc.h"

// Modulators
#include "blockmodem.h"
#include "mpsk.h"
#include "qam.h"
#include "dminner.h"
#include "dminner2.h"
#include "dminner2d.h"

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
#include "codec_reshaped.h"
#include "uncoded.h"
#include "mapcc.h"
#include "turbo.h"
#include "repacc.h"
#include "sysrepacc.h"

// Signal Mappers
#include "mapper.h"
#include "map_straight.h"
#include "map_interleaved.h"
#include "map_permuted.h"
#include "map_stipple.h"

// Systems
#include "commsys.h"
#include "commsys_iterative.h"

// Experiments
#include "commsys_simulator.h"
#include "commsys_prof_burst.h"
#include "commsys_prof_pos.h"
#include "commsys_prof_sym.h"
#include "commsys_hist_symerr.h"
#include "commsys_threshold.h"

#include <iostream>

namespace libcomm {

/*!
 \brief   Communications Library Serializer.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

// Serialization support
class serializer_libcomm : private qsc<libbase::gf<1, 0x3> > ,
      awgn,
      laplacian,
      lapgauss,
      bsid,
      bsid2d,
      bsc,
      nrcc,
      rscc,
      dvbcrsc,
      grscc<libbase::gf<1, 0x3> > ,
      gnrcc<libbase::gf<1, 0x3> > ,
      uncoded<double> ,
      mapcc<double> ,
      turbo<double> ,
      onetimepad<double> ,
      padded<double> ,
      berrou<double> ,
      flat<double> ,
      helical<double> ,
      rand_lut<double> ,
      rectangular<double> ,
      shift_lut<double> ,
      uniform_lut<double> ,
      named_lut<double> ,
      codec_reshaped<turbo<double> > {
private:
   typedef libbase::logrealfast logrealfast;
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
   // Modulators
   mpsk _mpsk;
   qam _qam;
   dminner<double, true> _dminner;
   dminner2<double, true> _dminner2;
   dminner2d<double, true> _dminner2d;
   // Codecs
   repacc<double> _repacc;
   sysrepacc<double> _sysrepacc;
   // Mappers
   map_interleaved<libbase::vector> _map_interleaved;
   map_permuted<libbase::vector> _map_permuted;
   map_stipple<libbase::vector> _map_stipple;
   // Systems
   commsys<bool> _commsys;
   commsys_iterative<bool> _commsys_iterative;
   // Experiments
   commsys_simulator<bool> _commsys_simulator;
   commsys_threshold<bool> _commsys_threshold;
public:
   serializer_libcomm() :
      _mpsk(2), _qam(4)
      {
      }
};

// Public interface to load objects

template <class T>
T *loadandverify(std::istream& file)
   {
   const serializer_libcomm my_serializer_libcomm;
   T *system;
   file >> system;
   libbase::verifycompleteload(file);
   return system;
   }

template <class T>
T *loadfromstring(const std::string& systemstring)
   {
   // load system from string representation
   std::istringstream is(systemstring, std::ios_base::in
         | std::ios_base::binary);
   return libcomm::loadandverify<T>(is);
   }

template <class T>
T *loadfromfile(const std::string& fname)
   {
   // load system from file
   std::ifstream file(fname.c_str(), std::ios_base::in | std::ios_base::binary);
   assertalways(file.is_open());
   return libcomm::loadandverify<T>(file);
   }

} // end namespace

#endif
