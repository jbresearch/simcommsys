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
#include <list>
#include <vector>
#include <string>

namespace libcomm {

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

class instrumented {
private:
   std::list<double> m_timings; //!< List of timings taken
   std::list<std::string> m_names; //!< List of friendly names

   // TODO: change back to protected!
public:
   /*! \name Interface for derived classes */
   //! Add a single timer (from components)
   void add_timer(double time, const std::string& name)
      {
      m_timings.push_back(time);
      m_names.push_back(name);
      }
   //! Add a single timer (from timer object, stopping timer if necessary)
   void add_timer(libbase::timer& timer)
      {
      if (timer.isrunning())
         timer.stop();
      m_timings.push_back(timer.elapsed());
      m_names.push_back(timer.get_name());
      }
   //! Batch add timers
   void add_timers(const instrumented& component)
      {
      m_timings.insert(m_timings.end(), component.m_timings.begin(),
            component.m_timings.end());
      m_names.insert(m_names.end(), component.m_names.begin(),
            component.m_names.end());
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   virtual ~instrumented()
      {
      }
   // @}

   /*! \name User interface */
   //! Clear list of timers
   void reset_timers()
      {
      m_timings.clear();
      m_names.clear();
      }
   //! Get the list of timings taken
   std::vector<double> get_timings() const
      {
      std::vector<double> result;
      result.assign(m_timings.begin(), m_timings.end());
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

} // end namespace

#endif
