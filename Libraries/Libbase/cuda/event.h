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

#ifndef __cuda_event_h
#define __cuda_event_h

#include "config.h"
#include "cuda-all.h"

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

#ifdef __CUDACC__

/*!
 * \brief   A CUDA event
 * \author  Johann Briffa
 *
 * This class represents an identifier for an event on a stream. An event has
 * occurred when all other preceding commands on the same stream have completed.
 * This can be used to set dependencies between streams.
 *
 * Note that this is a host-only object. Device code has no interface to events.
 */

class event {
protected:
   /*! \name Object representation */
   cudaEvent_t eid; //!< Event identifier
   // @}

private:
   /*! \name Law of the Big Three */
   /*! \brief Copy constructor
    * \note Copy construction is disabled as it has no meaning.
    */
   event(const event& x);
   /*! \brief Copy assignment operator
    * \note Copy assignment is disabled as it has no meaning.
    */
   event& operator=(const event& x);
   // @}

public:
   /*! \name Constructors */
   /*! \brief Default constructor
    * Creates and initializes event object.
    */
   event()
      {
      cudaSafeCall(cudaEventCreateWithFlags(&eid, cudaEventDisableTiming));
      }
   // @}

   /*! \name Law of the Big Three */
   /*! \brief Destructor
    * Destroys the event object.
    */
   ~event()
      {
      cudaSafeCall(cudaEventDestroy(eid));
      }
   // @}

   /*! \name User interface */
   //! Returns event identifier to use where needed
   const cudaEvent_t& get_id() const
      {
      return eid;
      }
   //! Waits for this event to complete
   void sync() const
      {
      cudaSafeCall(cudaEventSynchronize(eid));
      }
   //! Records an event on given stream
   void record(const stream& s) const
      {
      cudaSafeCall(cudaEventRecord(eid, s.get_id()));
      }
   //! Records an event on default stream
   void record() const
      {
      cudaSafeCall(cudaEventRecord(eid, 0));
      }
   // @}
};

#endif

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
