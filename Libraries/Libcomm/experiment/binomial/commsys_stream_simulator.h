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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __commsys_stream_simulator_h
#define __commsys_stream_simulator_h

#include "config.h"
#include "commsys_simulator.h"

namespace libcomm {

/*!
 * \brief   Communication System Simulator - for stream-oriented modulator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
 */
template <class S, class R = commsys_errorrates>
class commsys_stream_simulator : public commsys_simulator<S, R> {
private:
   // Shorthand for class hierarchy
   typedef experiment Interface;
   typedef commsys_stream_simulator<S, R> This;
   typedef commsys_simulator<S, R> Base;

private:
   /*! \name Internally-used objects */
   libbase::vector<int> source_this; //!< Message for current frame
   libbase::vector<int> source_next; //!< Message for next frame
   libbase::vector<S> received_prev; //!< Received sequence for previous frame
   libbase::vector<S> received_this; //!< Received sequence for current frame
   libbase::vector<S> received_next; //!< Received sequence for next frame
   libbase::vector<double> eof_post; //!< Centralized posterior probabilities at end-of-frame
   int drift_error; //!< Error in channel drift estimation at end-of-frame
   int cumulative_drift; //!< Actual cumulative channel drift at end-of-frame
   commsys<S>* sys_tx; //!< Copy of the commsys object for transmitter operations
   // @}

protected:
   /*! \name Setup functions */
   /*!
    * \brief Prepares to simulate a new sequence
    * This method clears the internal state and makes a copy of the base
    * commsys object to use for the transmission path.
    */
   void reset()
      {
      // Clear internal state
      source_this.init(0);
      source_next.init(0);
      received_prev.init(0);
      received_this.init(0);
      received_next.init(0);
      eof_post.init(0);
      drift_error = 0;
      cumulative_drift = 0;
      // Make a copy of the commsys object for transmitter operations
      delete sys_tx;
      if (this->sys)
         sys_tx = dynamic_cast<commsys<S>*> (this->sys->clone());
      else
         sys_tx = NULL;
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   commsys_stream_simulator(const commsys_stream_simulator<S, R>& c) :
      commsys_simulator<S, R> (c), sys_tx(NULL)
      {
      reset();
      }
   commsys_stream_simulator() :
      sys_tx(NULL)
      {
      reset();
      }
   virtual ~commsys_stream_simulator()
      {
      delete sys_tx;
      }
   // @}

   /*! \name Communication System Setup */
   void seedfrom(libbase::random& r)
      {
      Base::seedfrom(r);
      reset();
      }
   // @}

   // Experiment handling
   void sample(libbase::vector<double>& result);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(commsys_stream_simulator)
};

} // end namespace

#endif
