/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __serializer_libcomm_h
#define __serializer_libcomm_h

// Utilities
#include "gf.h"
#include "erasable.h"

// Arithmetic Types
#include "mpgnu.h"
#include "mpreal.h"
#include "logreal.h"
#include "logrealfast.h"

// Channels
#include "channel.h"
#include "channel/awgn.h"
#include "channel/laplacian.h"
#include "channel/lapgauss.h"
#include "channel/qids.h"
#include "channel/bpmr.h"
#include "channel/dids.h"
#include "channel/qsc.h"
#include "channel/qec.h"

// Embedders - atomic
#include "embedder.h"
#include "embedder/lsb.h"
#include "embedder/qim.h"
#include "embedder/ssis.h"
// Embedders - block
#include "blockembedder.h"
#include "embedder/direct_blockembedder.h"

// Modulators
#include "blockmodem.h"
#include "modem/direct_modem.h"
#include "modem/direct_blockmodem.h"
#include "modem/mpsk.h"
#include "modem/qam.h"
#include "modem/dminner.h"
#include "modem/tvb.h"
#include "modem/marker.h"

// Convolutional Encoders
#include "fsm.h"
#include "fsm/nrcc.h"
#include "fsm/rscc.h"
#include "fsm/dvbcrsc.h"
#include "fsm/grscc.h"
#include "fsm/gnrcc.h"
#include "fsm/zsm.h"

// Interleavers
#include "interleaver.h"
#include "interleaver/onetimepad.h"
#include "interleaver/padded.h"
// LUT Interleavers
#include "interleaver/lut_interleaver.h"
#include "interleaver/lut/berrou.h"
#include "interleaver/lut/flat.h"
#include "interleaver/lut/helical.h"
#include "interleaver/lut/rand_lut.h"
#include "interleaver/lut/rectangular.h"
#include "interleaver/lut/shift_lut.h"
#include "interleaver/lut/uniform_lut.h"
// Named LUT
#include "interleaver/lut/named_lut.h"
#include "interleaver/lut/named/file_lut.h"
#include "interleaver/lut/named/stream_lut.h"
#include "interleaver/lut/named/vale96int.h"

// Codecs
#include "codec.h"
#include "codec/codec_concatenated.h"
#include "codec/codec_multiblock.h"
#include "codec/codec_reshaped.h"
#include "codec/memoryless.h"
#include "codec/uncoded.h"
#include "codec/mapcc.h"
#include "codec/turbo.h"
#include "codec/repacc.h"
#include "codec/sysrepacc.h"
#include "codec/reedsolomon.h"
#include "codec/ldpc.h"

// Signal Mappers
#include "mapper.h"
#include "mapper/map_concatenated.h"
#include "mapper/map_straight.h"
#include "mapper/map_interleaved.h"
#include "mapper/map_permuted.h"
#include "mapper/map_punctured.h"
#include "mapper/map_stipple.h"
#include "mapper/map_aggregating.h"
#include "mapper/map_dividing.h"

// Systems
#include "commsys.h"
#include "commsys_iterative.h"
#include "commsys_fulliter.h"

// Experiments
#include "experiment/binomial/commsys_simulator.h"
#include "experiment/binomial/commsys_stream_simulator.h"
#include "experiment/binomial/commsys_threshold.h"
#include "experiment/normal/commsys_timer.h"
#include "experiment/normal/exit_computer.h"

#include <iostream>

namespace libcomm {

/*!
 * \brief   Communications Library Serializer.
 * \author  Johann Briffa
 */

// Serialization support
class serializer_libcomm : private
      nrcc,
      rscc,
      dvbcrsc,
      grscc<libbase::gf2> ,
      gnrcc<libbase::gf2> ,
      zsm<libbase::gf2> ,

      mapcc<double> ,

      onetimepad<double> ,
      padded<double> ,
      berrou<double> ,
      flat<double> ,
      helical<double> ,
      rand_lut<double> ,
      rectangular<double> ,
      shift_lut<double> ,
      uniform_lut<double> ,
      named_lut<double> {
private:
   typedef libbase::logrealfast logrealfast;
private:
   // Channels
   awgn _awgn;
   laplacian<sigspace> _laplacian;
   lapgauss _lapgauss;
   qsc<libbase::gf2> _qsc;
   qec<libbase::erasable<libbase::gf2> > _qec;
   qids<libbase::gf2, float> _qids;
   bpmr<float> _bpmr;
   dids<float> _dids;
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
   // Embedders - atomic
   qim<int> _qim_int;
   lsb<int> _lsb_int;
   // Embedders - block
   direct_blockembedder<int> _direct_blockembedder_int;
   ssis<int> _ssis_int;
   // Modulators
   direct_modem<bool> _direct_modem_bool;
   direct_blockmodem<bool> _direct_blockmodem_bool;
   mpsk _mpsk;
   qam _qam;
   dminner<double> _dminner;
   tvb<bool, double, float> _tvb;
   marker<bool, double, float> _marker;
   // Convolutional Encoders
   //nrcc _nrcc;
   //rscc _rscc;
   //dvbcrsc _dvbcrsc;
   //grscc<libbase::gf2> _grscc;
   //gnrcc<libbase::gf2> _gnrcc;
   //zsm<libbase::gf2> _zsm;
   // Codecs
   codec_concatenated<libbase::vector, double> _codec_concatenated;
   codec_multiblock<libbase::vector, double> _codec_multiblock;
   memoryless<double> _memoryless;
   uncoded<double> _uncoded;
   //mapcc<double> _mapcc;
   turbo<double, double> _turbo;
   codec_reshaped<turbo<double, double> > _reshaped_turbo;
   ldpc<libbase::gf2, double> _ldpc;
   reedsolomon<libbase::gf8> _reedsolomon;
   repacc<double> _repacc;
   sysrepacc<double> _sysrepacc;
   // Mappers
   map_concatenated<libbase::vector> _map_concatenated;
   map_straight<libbase::vector> _map_straight;
   map_interleaved<libbase::vector> _map_interleaved;
   map_permuted<libbase::vector> _map_permuted;
   map_punctured<libbase::vector> _map_punctured;
   map_stipple<libbase::vector> _map_stipple;
   map_aggregating<libbase::vector> _map_aggregating;
   map_dividing<libbase::vector> _map_dividing;
   // Systems
   commsys<bool> _commsys;
   commsys_iterative<bool> _commsys_iterative;
   commsys_fulliter<bool> _commsys_fulliter;
   // Experiments
   commsys_simulator<bool, errors_hamming> _commsys_simulator;
   commsys_stream_simulator<bool, errors_hamming, float> _commsys_stream_simulator;
   commsys_threshold<bool, errors_hamming> _commsys_threshold;
   commsys_timer<bool> _commsys_timer;
   exit_computer<bool> _exit_computer;
public:
   serializer_libcomm() :
      _mpsk(2), _qam(4)
      {
      }
};

// Public interface to load objects

template <class T>
T *loadandverify(std::istream& sin)
   {
   const serializer_libcomm my_serializer_libcomm;
   T *system;
   sin >> libbase::eatcomments >> system >> libbase::verifycomplete;
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
