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

#ifndef __walltimer_h
#define __walltimer_h

#include "timer.h"

#include <ctime>
#ifdef WIN32
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
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * A class implementing a wall-clock timer; resolution is in microseconds
 * on UNIX and milliseconds on Win32.
 *
 * \todo Extract common base class for walltimer and cputimer
 */

class walltimer : public timer {
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
      struct _timeb tb;
      _ftime(&tb);
      return (double) tb.time + (double) tb.millitm * 1E-3;
#else
      struct timeval tv;
      struct timezone tz;
      gettimeofday(&tv, &tz);
      return (double) tv.tv_sec + (double) tv.tv_usec * 1E-6;
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
};

} // end namespace

#endif
