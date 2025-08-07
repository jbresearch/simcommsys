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

#ifndef __instrumented_h
#define __instrumented_h

#include "config.h"
#include "timer.h"
#include <algorithm>
#include <list>
#include <string>
#include <vector>

namespace libcomm
{

/*!
 * \brief   Instrumented Class Interface.
 * \author  Johann Briffa
 *
 * Defines a class that is instrumented for internal timings.
 * Classes that inherit this public interface need to call reset() at the
 * start of a cycle of timed events and add a timer for each timed event
 * within the cycle. This class also provides an interface for bulk addition
 * of timers, to facilitate implementation in classes that contain other
 * instrumented classes.
 */

class instrumented
{
private:
    std::list<size_t>
        m_counts; //!< List of number of timings with particular time taken
    std::list<double> m_timings;    //!< List of timings taken
    std::list<std::string> m_names; //!< List of friendly names

    // TODO: change back to protected!
public:
    //! Add a single timer (from components)
    void add_timer(double time, const std::string& name)
    {
        m_timings.push_back(time);
        m_names.push_back(name);
        m_counts.push_back(1);
    }
    //! Add a single timer (from timer object, stopping timer if necessary)
    void add_timer(libbase::timer& timer)
    {
        if (timer.isrunning()) {
            timer.stop();
        }

        m_timings.push_back(timer.elapsed());
        m_names.push_back(timer.get_name());
        m_counts.push_back(1);
    }
    //! Add a single timer (from components) with variance
    void add_timer_with_variance(double time, const std::string& name)
    {
        m_timings.push_back(time);
        m_names.push_back(name);
        m_counts.push_back(1);
        // add variance
        m_timings.push_back(time * time);
        m_names.push_back(name + "_sq");
        m_counts.push_back(1);
    }
    //! Add a single, new timer, or if a timer with same name already exists,
    //! accumulate
    void add_or_accumulate_timer(libbase::timer& timer)
    {
        if (timer.isrunning()) {
            timer.stop();
        }

        auto m_names_it = m_names.begin();
        auto m_counts_it = m_counts.begin();
        auto m_timings_it = m_timings.begin();
        bool found = false; // was a timing with the same name found?
        for (; m_names_it != m_names.end();
             ++m_names_it, ++m_counts_it, ++m_timings_it) {
            if (*m_names_it == timer.get_name()) {
                ++*m_counts_it;
                *m_timings_it += timer.elapsed();
                found = true;
                break;
            }
        }
        if (!found) { // this is a new timer
            m_timings.push_back(timer.elapsed());
            m_names.push_back(timer.get_name());
            m_counts.push_back(1);
        }
    }
    //! Add a single, new timer, or if a timer with same name already exists,
    //! accumulate
    //! Also add square of timing so that variance in timing result can be
    //! computed
    void add_or_accumulate_timer_with_variance(libbase::timer& timer)
    {
        add_or_accumulate_timer(timer);
        add_or_accumulate_timer(timer.elapsed() * timer.elapsed(),
                                timer.get_name() + "_sq");
    }
    //! Add a single, new timing, or if a timing with same name already exists,
    //! accumulate
    void
    add_or_accumulate_timer(double time, const std::string& name)
    {
        auto m_names_it = m_names.begin();
        auto m_counts_it = m_counts.begin();
        auto m_timings_it = m_timings.begin();
        bool found = false; // was a timing with the same name found?
        for (; m_names_it != m_names.end();
             ++m_names_it, ++m_counts_it, ++m_timings_it) {
            if (*m_names_it == name) {
                ++*m_counts_it;
                *m_timings_it += time;
                found = true;
                break;
            }
        }
        if (!found) { // this is a new timer
            m_timings.push_back(time);
            m_names.push_back(name);
            m_counts.push_back(1);
        }
    }
    //! Batch add timers
    void add_timers(const instrumented& component)
    {
        m_timings.insert(m_timings.end(),
                         component.m_timings.begin(),
                         component.m_timings.end());
        m_names.insert(
            m_names.end(), component.m_names.begin(), component.m_names.end());
        m_counts.insert(m_counts.end(),
                        component.m_counts.begin(),
                        component.m_counts.end());
    }
    //! Batch add or accumulate timers
    void add_or_accumulate_timers(const instrumented& component)
    {
        auto m_timings_it = component.m_timings.begin();
        auto m_names_it = component.m_names.begin();
        auto m_counts_it = component.m_counts.begin();
        for (; m_timings_it != component.m_timings.end();
             ++m_timings_it, ++m_names_it, ++m_counts_it) {
            this->add_or_accumulate_timer(
                *m_timings_it, *m_names_it, *m_counts_it);
        }
    }
    //! Initialize a timing with a certain name. Both the count and total time
    //! will be set to 0.
    void init_timer(std::string name)
    {
        m_timings.push_back(0);
        m_names.push_back(name);
        m_counts.push_back(0);
    }
    //! Initialize a timing with a certain name, and its squared version. Both
    //! the count and total time will be set to 0.
    void init_timer_with_variance(std::string name)
    {
        init_timer(name);
        init_timer(name + "_sq");
    }
    // @}

public:
    /*! \name Constructors / Destructors */
    virtual ~instrumented() {}
    // @}

    /*! \name User interface */
    //! Clear list of timers
    void reset_timers()
    {
        m_timings.clear();
        m_names.clear();
        m_counts.clear();
    }
    //! Get the list of timings taken
    std::vector<double> get_timings() const
    {
        std::vector<double> result;
        auto cnt = m_counts.begin();
        for (auto res = m_timings.begin(); res != m_timings.end();
             ++res, ++cnt) {
            result.push_back(*cnt == 0 ? 0 : *res / *cnt);
        }
        return result;
    }
    //! Get the list of friendly names for timings taken
    std::vector<std::string> get_names() const
    {
        std::vector<std::string> result;
        result.assign(m_names.begin(), m_names.end());
        return result;
    }
    // @}
};

} // namespace libcomm

#endif
