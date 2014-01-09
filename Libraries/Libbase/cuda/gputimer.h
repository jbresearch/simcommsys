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

#ifndef __cuda_gputimer_h
#define __cuda_gputimer_h

#include "../timer.h"

namespace cuda {

/*!
 * \brief   GPU Timer.
 * \author  Johann Briffa
 *
 * A class which can be used to time subroutines, etc. using timers on the
 * GPU itself. Resolution is about half a microsecond.
 */

class gputimer : public libbase::timer {
private:
   /*! \name Internal representation */
   cudaEvent_t event_start; //!< Start event object on GPU
   mutable cudaEvent_t event_stop; //!< Stop event object on GPU
   cudaStream_t stream; //!< Stream on which timer operates
   // @}

protected:
   /*! \name Interface with derived class */
   void do_start()
      {
      cudaSafeCall(cudaEventRecord(event_start, stream));
      }
   void do_stop() const
      {
      cudaSafeCall(cudaEventRecord(event_stop, stream));
      }
   double get_elapsed() const
      {
      // determine time difference between start and stop events, in milli-sec
      float time;
      cudaSafeCall(cudaEventSynchronize(event_stop));
      cudaSafeCall(cudaEventElapsedTime(&time, event_start, event_stop));
      return time * 1e-3;
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Main constructor
   explicit gputimer(const std::string& name = "", const cudaStream_t stream =
         0, const bool running = true) :
      timer(name), stream(stream)
      {
      cudaSafeCall(cudaEventCreate(&event_start));
      cudaSafeCall(cudaEventCreate(&event_stop));
      init(running);
      }
   //! Destructor
   ~gputimer()
      {
      expire();
      cudaSafeCall(cudaEventDestroy(event_start));
      cudaSafeCall(cudaEventDestroy(event_stop));
      }
   // @}

   /*! \name GPU-specific methods */
   void set_stream(const cudaStream_t stream)
      {
      this->stream = stream;
      }
   // @}

   /*! \name Timer information */
   double resolution() const
      {
      // Note: this resolution is approximate, as given in CUDA docs
      return 0.5e-6;
      }
   // @}
};

} // end namespace

#endif
