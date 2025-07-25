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
#include <sys/time.h>

namespace libbase
{

/*!
 * \brief   Wallclock Timer.
 * \author  Johann Briffa
 *
 * A class implementing a wall-clock timer; resolution is in microseconds
 * on UNIX.
 *
 * \todo Extract common base class for walltimer and cputimer
 */

class walltimer : public timer
{
private:
    /*! \name Internal representation */
    struct timeval event_start;        //!< Start event time object
    mutable struct timeval event_stop; //!< Stop event time object
    // @}

private:
    /*! \name Internal helper methods */
    static double convert(const struct timeval& tv)
    {
        return tv.tv_sec + double(tv.tv_usec) * 1E-6;
    }
    // @}

protected:
    /*! \name Interface with derived class */
    void do_start()
    {
        struct timezone tz;
        gettimeofday(&event_start, &tz);
    }
    void do_stop() const
    {
        struct timezone tz;
        gettimeofday(&event_stop, &tz);
    }
    double get_elapsed() const
    {
        return convert(event_stop) - convert(event_start);
    }
    // @}

public:
    /*! \name Constructors / Destructors */
    //! Main constructor
    explicit walltimer(const std::string& name = "", const bool running = true)
        : timer(name)
    {
        init(running);
    }
    //! Destructor
    ~walltimer() { expire(); }
    // @}

    /*! \name Timer information */
    double resolution() const { return 1e-6; }
    // @}
};

} // namespace libbase

#endif
