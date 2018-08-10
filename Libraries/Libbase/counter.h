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

#ifndef __counter_h
#define __counter_h

#include "config.h"
#include <string>
#include <iostream>

namespace libbase {

/*!
 * \brief   Counter class
 * \author  Johann Briffa
 *
 * A class for counters which can be used to count events, etc.
 */

class counter {
private:
   /*! \name Internal representation */
   std::string name; //!< Counter name
   size_t count; //!< Internal count
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   counter(const std::string& name = "") :
      name(name), count(0)
      {
      }
   //! Virtual destructor
   virtual ~counter()
      {
      std::clog << "Counter";
      if (name != "")
         std::clog << " (" << name << ")";
      std::clog << ": " << count << " events" << std::endl;
      }
   // @}

   /*! \name Counter operation */
   //! Resets the counter
   void reset()
      {
      count = 0;
      }
   //! Increments the counter
   void increment()
      {
      count++;
      }
   // @}

   /*! \name Timer information */
   //! Return current count
   size_t get_count() const
      {
      return count;
      }
   //! Return counter name
   std::string get_name() const
      {
      return name;
      }
   // @}
};

} // end namespace

#endif
