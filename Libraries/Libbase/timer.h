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

#ifndef __timer_h
#define __timer_h

#include "config.h"
#include <string>
#include <iostream>

namespace libbase {

/*!
 * \brief   Timer - base class
 * \author  Johann Briffa
 *
 * A base class for timers which can be used to time subroutines, etc.
 * Time is expressed in seconds and resolution depends on implementation;
 * range allows durations in weeks/years/anything you care to time...
 */

class timer {
private:
   /*! \name Internal representation */
   std::string name; //!< Timer name
   bool running; //!< Flag indicating if timer is currently running
   bool valid; //!< Flag indicating if timer was started at least once
   // @}

protected:
   /*! \name Interface with derived class */
   /*! \brief Initialization
    * By default, the timer starts running on initialization.
    */
   void init(const bool running = true)
      {
      if (running)
         start();
      }
   /*! \brief End-of-life reached
    * If a timer is running, a message is output to the standard log,
    * indicating expiry and the timer's duration.
    */
   void expire();
   /*! \brief Start the timer, keeping track of start event
    * This method may be called more than once, over-writing the event record.
    */
   virtual void do_start() = 0;
   /*! \brief Stop the timer, keeping track of stop event
    * This method may be called more than once, over-writing the event record.
    * Note that marking a stop event does not constitute a visible change.
    */
   virtual void do_stop() const = 0;
   /*! \brief Determine time elapsed from start to stop event
    * This method will only be called after both start and stop methods have
    * been called.
    */
   virtual double get_elapsed() const = 0;
   // @}

public:
   /*! \name Utility functions */
   //! The current date and time.
   static std::string date();
   //! Format the given time (in seconds) as a pretty string
   static std::string format(const double time);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   timer(const std::string& name = "") :
      name(name), running(false), valid(false)
      {
      }
   //! Virtual destructor
   virtual ~timer()
      {
      assert(!valid);
      assert(!running);
      }
   // @}

   /*! \name Timer operation */
   //! Starts the timer running; or if timer was running, restarts silently
   void start()
      {
      running = true;
      valid = true;
      do_start();
      }
   //! Stops the timer; causes an error if called on a stopped timer
   void stop()
      {
      assert(running);
      do_stop();
      running = false;
      }
   // @}

   /*! \name Timer information */
   /*! \brief The number of seconds elapsed.
    * If the timer is not running, this returns the number of seconds between
    * the timer start and stop events.
    * If the timer is running, this returns the number of seconds since the
    * timer was started.
    * \note It is an error to call this on a timer that was never started.
    */
   double elapsed() const
      {
      assert(valid);
      if (running)
         do_stop();
      return get_elapsed();
      }
   //! Return the timer resolution in seconds
   virtual double resolution() const = 0;
   //! Return true if timer is currently running
   bool isrunning() const
      {
      return running;
      }
   //! Return timer name
   std::string get_name() const
      {
      return name;
      }
   // @}

   /*! \name Conversion operations */
   //! Conversion function to generate a string
   operator std::string() const
      {
      return format(elapsed());
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
