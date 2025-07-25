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

#ifndef __cputimer_h
#define __cputimer_h

#include "timer.h"

#include <ctime>

namespace libbase
{

/*!
 * \brief   CPU-usage Timer.
 * \author  Johann Briffa
 *
 * A class implementing a CPU-usage timer; this keeps track of time used by the
 * process (including any threads, but no children); uses the hardware
 * high-precision even timer, so resolution is sub-microsecond on all systems.
 *
 * \todo Extract common base class for walltimer and cputimer
 */

class cputimer : public timer
{
private:
    /*! \name Internal representation */
    struct timespec event_start;        //!< Start event usage info
    mutable struct timespec event_stop; //!< Stop event usage info
    // @}

private:
    /*! \name Internal helper methods */
    static double convert(const struct timespec& tv)
    {
        return tv.tv_sec + double(tv.tv_nsec) * 1E-9;
    }
    // @}

protected:
    /*! \name Interface with derived class */
    void do_start() { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &event_start); }
    void do_stop() const
    {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &event_stop);
    }
    double get_elapsed() const
    {
        return convert(event_stop) - convert(event_start);
    }
    // @}

public:
    /*! \name Constructors / Destructors */
    //! Main constructor
    explicit cputimer(const std::string& name = "", const bool running = true)
        : timer(name)
    {
        init(running);
    }
    //! Destructor
    ~cputimer() { expire(); }
    // @}

    /*! \name Timer information */
    double resolution() const
    {
        struct timespec res;
        clock_getres(CLOCK_PROCESS_CPUTIME_ID, &res);
        return convert(res);
    }
    // @}
};

} // namespace libbase

#endif
