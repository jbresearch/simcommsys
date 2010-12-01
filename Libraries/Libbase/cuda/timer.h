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

#ifndef __cuda_timer_h
#define __cuda_timer_h

#include "../timer.h"

namespace cuda {

/*!
 * \brief   GPU Timer.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * A class which can be used to time subroutines, etc. using timers on the
 * GPU itself. Resolution is about half a microsecond, but the class can also
 * handle durations in weeks/years/anything you care to time...
 */

class timer {
   std::string name;
   cudaEvent_t event_start, event_stop;
   bool running;
public:
   /*! \name Constructors / Destructors */
   explicit timer(const std::string& name = "") :
      name(name), running(false)
      {
      cudaEventCreate(&event_start);
      cudaEventCreate(&event_stop);
      }
   virtual ~timer()
      {
      cudaEventDestroy(event_start);
      cudaEventDestroy(event_stop);
      }
   // @}

   /*! \name Timer operation */
   void start()
      {
      cudaEventRecord(event_start, 0);
      running = true;
      }
   void stop()
      {
      assert(running);
      cudaEventRecord(event_stop, 0);
      running = false;
      }
   // @}

   /*! \name Timer information */
   //! The number of seconds elapsed.
   double elapsed() const
      {
      if (running)
         cudaEventRecord(event_stop, 0);
      // determine time difference between start and stop events, in milli-sec
      float time;
      cudaEventSynchronize(event_stop);
      cudaEventElapsedTime(&time, event_start, event_stop);
      return time * 1e-3;
      }
   bool isrunning() const
      {
      return running;
      }
   // @}

   /*! \name Conversion operations */
   //! Conversion function to generate a string
   operator std::string() const
      {
      return libbase::timer::format(elapsed());
      }
   // @}
};

/*!
 * \brief Stream output
 *
 * \note This routine does not stop the timer, therefore allowing
 * display of running timers.
 */
inline std::ostream& operator<<(std::ostream& s, const timer& t)
   {
   return s << std::string(t);
   }

} // end namespace

#endif
