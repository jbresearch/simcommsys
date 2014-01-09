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

#ifndef __walltimer_h
#define __walltimer_h

#include "timer.h"

#include <ctime>
#ifdef _WIN32
#  include <sys/types.h>
#  include <sys/timeb.h>
#else
#  include <sys/time.h>
#endif

namespace libbase {

/*!
 * \brief   Wallclock Timer.
 * \author  Johann Briffa
 *
 * A class implementing a wall-clock timer; resolution is in microseconds
 * on UNIX and milliseconds on Win32.
 *
 * \todo Extract common base class for walltimer and cputimer
 */

class walltimer : public timer {
private:
   /*! \name Internal representation */
#ifdef _WIN32
   struct _timeb event_start; //!< Start event time object
   mutable struct _timeb event_stop; //!< Stop event time object
#else
   struct timeval event_start; //!< Start event time object
   mutable struct timeval event_stop; //!< Stop event time object
#endif
   // @}

private:
   /*! \name Internal helper methods */
#ifdef _WIN32
   static double convert(const struct _timeb& tb)
      {
      return tb.time + double(tb.millitm) * 1E-3;
      }
#else
   static double convert(const struct timeval& tv)
      {
      return tv.tv_sec + double(tv.tv_usec) * 1E-6;
      }
#endif
   // @}

protected:
   /*! \name Interface with derived class */
   void do_start()
      {
#ifdef _WIN32
      _ftime(&event_start);
#else
      struct timezone tz;
      gettimeofday(&event_start, &tz);
#endif
      }
   void do_stop() const
      {
#ifdef _WIN32
      _ftime(&event_stop);
#else
      struct timezone tz;
      gettimeofday(&event_stop, &tz);
#endif
      }
   double get_elapsed() const
      {
      return convert(event_stop) - convert(event_start);
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Main constructor
   explicit walltimer(const std::string& name = "", const bool running = true) :
      timer(name)
      {
      init(running);
      }
   //! Destructor
   ~walltimer()
      {
      expire();
      }
   // @}

   /*! \name Timer information */
   double resolution() const
      {
#ifdef _WIN32
      return 1e-3;
#else
      return 1e-6;
#endif
      }
   // @}
};

} // end namespace

#endif
