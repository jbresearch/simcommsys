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

#ifndef __cputimer_h
#define __cputimer_h

#include "timer.h"

#include <ctime>
#ifdef WIN32
// NOTE: the following line avoids problems with including winsock2.h later
#  define _WINSOCKAPI_
#  include <windows.h>
#else
#  include <sys/resource.h>
#endif

namespace libbase {

/*!
 * \brief   CPU-usage Timer.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * A class implementing a CPU-usage timer; this keeps track of all (user +
 * system) time used by the process (including any threads, but no children);
 * resolution is in microseconds on UNIX and sub-microsecond on Win32 systems.
 *
 * \todo Extract common base class for walltimer and cputimer
 */

class cputimer : public timer {
private:
   /*! \name Internal representation */
#ifdef WIN32
   LARGE_INTEGER event_start; //!< Start event usage info
   mutable LARGE_INTEGER event_stop; //!< Stop event usage info
#else
   struct rusage event_start; //!< Start event usage info
   mutable struct rusage event_stop; //!< Stop event usage info
#endif
   // @}

private:
   /*! \name Internal helper methods */
#ifdef WIN32
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
#ifdef WIN32
      QueryPerformanceCounter(&event_start);
#else
      getrusage(RUSAGE_SELF, &event_start);
#endif
      }
   void do_stop() const
      {
#ifdef WIN32
      QueryPerformanceCounter(&event_stop);
#else
      getrusage(RUSAGE_SELF, &event_stop);
#endif
      }
   double get_elapsed() const
      {
#ifdef WIN32
      // get ticks per second
      LARGE_INTEGER frequency;
      QueryPerformanceFrequency(&frequency);
      return double(event_stop.QuadPart - event_start.QuadPart) / double(frequency.QuadPart);
#else
      const double utime = convert(event_stop.ru_utime) - convert(
            event_start.ru_utime);
      const double stime = convert(event_stop.ru_stime) - convert(
            event_start.ru_stime);
      return utime + stime;
#endif
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Main constructor
   explicit cputimer(const std::string& name = "", const bool running = true) :
      timer(name)
      {
      init(running);
      }
   //! Destructor
   ~cputimer()
      {
      expire();
      }
   // @}

   /*! \name Timer information */
   double resolution() const
      {
#ifdef WIN32
      // get ticks per second
      LARGE_INTEGER frequency;
      QueryPerformanceFrequency(&frequency);
      return 1.0 / double(frequency.QuadPart);
#else
      return 1e-6;
#endif
      }
   // @}
};

} // end namespace

#endif
