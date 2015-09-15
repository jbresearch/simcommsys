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

#ifndef __commsys_stream_simulator_h
#define __commsys_stream_simulator_h

#include "config.h"
#include "commsys_simulator.h"
#include "commsys_stream.h"
#include "result_collector/commsys/fidelity_pos.h"
#include "hard_decision.h"
#include <list>

namespace libcomm {

/*!
 * \brief   Communication System Simulator - for stream-oriented modulator.
 * \author  Johann Briffa
 *
 * A variation on the regular commsys_simulator object, making use of the
 * stream-oriented modulator interface additions.
 *
 * This simulates stream transmission and reception by keeping track of the
 * previous and next frame information. Except for the first frame (where the
 * start position is known exactly) the a-priori information for start-of-frame
 * position is set to the posterior information of the end-of-frame from the
 * previous frame simulation. The a-priori end-of-frame information is set
 * according to the distribution provided by the channel.
 *
 * \tparam S Channel symbol type
 * \tparam R Results collector type
 * \tparam real Floating-point type for metric computer interface
 */
template <class S, class R, class real>
class commsys_stream_simulator : public commsys_simulator<S, R> {
private:
   // Shorthand for class hierarchy
   typedef commsys_stream_simulator<S, R, real> This;
   typedef commsys_simulator<S, R> Base;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<S> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

private:
   /*! \name User-defined parameters */
   enum stream_mode_enum {
      stream_mode_open = 0, //!< Open-ended (non-terminating) stream
      stream_mode_reset, //!< Open-ended stream with reset after N frames
      stream_mode_terminated, //!< Stream of length N frames
      stream_mode_undefined
   } stream_mode; //!< enum indicating streaming mode
   int N; //!< number of frames to reset or end of stream
   // @}
   /*! \name Internally-used objects */
   std::list<array1i_t> source; //!< List of message sequences in order of transmission
   array1s_t received; //!< Received sequence as a stream
   array1d_t eof_post; //!< Centralized posterior probabilities at end-of-frame
   libbase::size_type<libbase::vector> offset; //!< Index offset for eof_post
   libbase::size_type<libbase::vector> estimated_drift; //!< Estimated drift in last decoded frame
   std::list<array1i_t> act_bdry_drift; //!< Actual channel drift at codeword boundaries of each received frame
   std::list<int> actual_drift; //!< Actual channel drift at end of each received frame
   int drift_error; //!< Cumulative error in channel drift estimation at end of last decoded frame
   commsys_stream<S, libbase::vector, real>* sys_enc; //!< Copy of the commsys object for encoder operations
   int frames_encoded; //!< Number of frames encoded since stream reset
   int frames_decoded; //!< Number of frames decoded since stream reset
   hard_decision<libbase::vector, double, int> hd_functor; //!< Hard-decision box
   // @}

protected:
   /*! \name Stream Extensions */
   //! Get communication system in stream mode
   commsys_stream<S, libbase::vector, real>& getsys_stream() const
      {
      return dynamic_cast<commsys_stream<S, libbase::vector, real>&>(*this->sys);
      }
   // @}

   /*! \name Setup functions */
   /*!
    * \brief Prepares to simulate a new sequence
    * This method clears the internal state and makes a copy of the base
    * commsys object to use for the transmission path.
    */
   void reset()
      {
      // clear internal state
      source.clear();
      received.init(0);
      eof_post.init(0);
      offset = libbase::size_type<libbase::vector>(0);
      estimated_drift = libbase::size_type<libbase::vector>(0);
      // reset drift trackers
      act_bdry_drift.clear();
      actual_drift.clear();
      drift_error = 0;
      // Make a copy of the commsys object for transmitter operations
      delete sys_enc;
      if (this->sys)
         {
         sys_enc = dynamic_cast<commsys_stream<S, libbase::vector, real>*>(this->sys->clone());
         assertalways(sys_enc);
         }
      else
         sys_enc = NULL;
      // reset counters
      frames_encoded = 0;
      frames_decoded = 0;
      }
   // @}

   // System Interface for Results
   int get_symbolsperframe() const
      {
      return sys_enc->getmodem()->input_block_size();
      }
   int get_symbolsperblock() const
      {
      return Base::get_symbolsperblock();
      }

public:
   /*! \name Constructors / Destructors */
   commsys_stream_simulator(const commsys_stream_simulator<S, R, real>& c) :
         commsys_simulator<S, R>(c), stream_mode(c.stream_mode), N(c.N), source(c.source), received(
               c.received), eof_post(c.eof_post), offset(c.offset), estimated_drift(
               c.estimated_drift), act_bdry_drift(c.act_bdry_drift), actual_drift(
               c.actual_drift), drift_error(c.drift_error), frames_encoded(
               c.frames_encoded), frames_decoded(c.frames_decoded)
      {
      sys_enc = dynamic_cast<commsys_stream<S, libbase::vector, real>*>(c.sys_enc->clone());
      }
   commsys_stream_simulator() :
         stream_mode(stream_mode_open), N(0), sys_enc(NULL)
      {
      reset();
      }
   virtual ~commsys_stream_simulator()
      {
      delete sys_enc;
      }
   // @}

   /*! \name Communication System Setup */
   void seedfrom(libbase::random& r)
      {
      // Call base method first
      Base::seedfrom(r);
      // Seed hard-decision box
      hd_functor.seedfrom(r);
      // Clear internal state
      reset();
      }
   void set_parameter(const double x)
      {
      Base::set_parameter(x);
      // we should already have a copy at this point
      assert(sys_enc);
      // set the TX channel parameter only (we should not need to use the RX)
      sys_enc->gettxchan()->set_parameter(x);
      }
   // @}

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const
      {
      // Get access to the results collector in codeword boundary analysis mode
      const fidelity_pos* rc = dynamic_cast<const fidelity_pos*>(this);
      const int base_count = (rc) ? R::count() : Base::count();
      return base_count * getsys_stream().sys_iter();
      }

   //! Clear list of timers
   void reset_timers()
      {
      Base::reset_timers();
      sys_enc->reset_timers();
      }

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER (commsys_stream_simulator)
};

} // end namespace

#endif
