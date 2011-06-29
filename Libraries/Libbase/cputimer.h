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
#ifndef WIN32
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
 * A class implementing a CPU-usage timer; resolution is in microseconds
 * on UNIX and CPU-ticks on Win32 (usually centi-seconds).
 *
 * \note On Win32, this is implemented using clock(), which is not a good
 *      approximation for CPU usage.
 *
 * \todo Re-implement on Win32 using more appropriate timers.
 *
 * \todo Extract common base class for walltimer and cputimer
 */

class cputimer : public timer {
private:
   /*! \name Internal representation */
   double event_start; //!< Start event time
   mutable double event_stop; //!< Stop event time
   // @}

private:
   /*! \name Internal helper methods */
   static double get_time()
      {
#ifdef WIN32
      return clock() / double(CLOCKS_PER_SEC);
#else
      struct rusage usage;
      double cpu;

      getrusage(RUSAGE_SELF, &usage);
      cpu = (double) usage.ru_utime.tv_sec + (double) usage.ru_utime.tv_usec
            * 1E-6;
      getrusage(RUSAGE_CHILDREN, &usage);
      cpu += (double) usage.ru_utime.tv_sec + (double) usage.ru_utime.tv_usec
            * 1E-6;

      return cpu;
#endif
      }
   // @}

protected:
   /*! \name Interface with derived class */
   void do_start()
      {
      event_start = get_time();
      }
   void do_stop() const
      {
      event_stop = get_time();
      }
   double get_elapsed() const
      {
      return event_stop - event_start;
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
};

} // end namespace

#endif
