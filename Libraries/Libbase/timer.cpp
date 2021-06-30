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

#include "timer.h"
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace libbase
{

// Utility functions

std::string timer::format(const double time)
{
    std::ostringstream sout;

    if (time < 60) {
        int order = int(ceil(-log10(time) / 3.0));
        if (order > 3) {
            order = 3;
        }

        sout << std::fixed << std::setprecision(2)
             << time * pow(10.0, order * 3);

        switch (order) {
        case 0:
            sout << "s";
            break;
        case 1:
            sout << "ms";
            break;
        case 2:
            sout << "us";
            break;
        case 3:
            sout << "ns";
            break;
        }
    } else {
        int days, hrs, min, sec;

        sec = int(floor(time));
        min = sec / 60;
        sec = sec % 60;
        hrs = min / 60;
        min = min % 60;
        days = hrs / 24;
        hrs = hrs % 24;
        if (days > 0) {
            sout << days;
            sout << (days == 1 ? " day, " : " days, ");
        }
        sout << std::setfill('0') << std::setw(2) << hrs << ':';
        sout << std::setfill('0') << std::setw(2) << min << ':';
        sout << std::setfill('0') << std::setw(2) << sec;
    }

    return sout.str();
}

std::string timer::date()
{
    const int max = 256;
    static char d[max];

    time_t t1 = time(NULL);
    struct tm* t2 = localtime(&t1);
    strftime(d, max, "%d %b %Y, %H:%M:%S", t2);

    return d;
}

// Interface with derived class

void timer::expire()
{
    if (running) {
        stop();
        std::clog << "Timer";

        if (name != "") {
            std::clog << " (" << name << ")";
        }

        std::clog << " expired after " << *this << std::endl;
    }

    // Invalidate to indicate we're using this properly
    assert(!running);
    valid = false;
}

} // namespace libbase
