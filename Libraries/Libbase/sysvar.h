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

#ifndef __sysvar_h
#define __sysvar_h

#include "config.h"
#include <string>
#include <sstream>

namespace libbase {

/*!
 * \brief   System variable.
 * \author  Johann Briffa
 *
 * Encapsulates a system environment variable, facilitating reading its value
 * as a string. Also provides automatic conversion to int and float.
 */

class sysvar {
private:
   /*! \name Internal representation */
   std::string name; //!< Variable name
   std::string value; //!< Variable contents as a string
   bool defined; //!< Flag indicating the variable is defined
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Main constructor
   explicit sysvar(const std::string& name) :
      name(name)
      {
      const char *s = getenv(name.c_str());
      if (s == NULL)
         defined = false;
      else
         {
         value = s;
         defined = true;
         }
      }
   // @}

   /*! \name Information */
   bool is_defined() const
      {
      return defined;
      }
   // @}

   /*! \name Explicit Conversion */
   //! Return value as string
   std::string as_string() const
      {
      assert(defined);
      return value;
      }
   //! Return value as integer
   int as_int() const
      {
      assert(defined);
      int r;
      std::istringstream(value) >> r;
      return r;
      }
   //! Return value as float
   double as_double() const
      {
      assert(defined);
      double r;
      std::istringstream(value) >> r;
      return r;
      }
   // @}

   /*! \name Automatic Conversion */
   //! Return value as string
   operator std::string() const
      {
      return as_string();
      }
   //! Return value as integer
   operator int() const
      {
      return as_int();
      }
   //! Return value as float
   operator double() const
      {
      return as_double();
      }
   // @}
};

} // end namespace

#endif
