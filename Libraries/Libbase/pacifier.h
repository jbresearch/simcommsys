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

#ifndef __pacifier_h
#define __pacifier_h

#include "config.h"
#include "walltimer.h"
#include <string>

namespace libbase {

/*!
 * \brief   User Pacifier.
 * \author  Johann Briffa
 *
 * A class that formats output suitable for keeping user updated with the
 * progress of an operation.
 */

class pacifier {
private:
   static bool quiet;
public:
   /*! \name Static interface */
   static void enable_output()
      {
      quiet = false;
      }
   static void disable_output()
      {
      quiet = true;
      }
   // @}

private:
   std::string name;
   walltimer t;
   int last;
   size_t characters;
public:
   /*! \name Constructors / Destructors */
   explicit pacifier(const std::string& name = "Process");
   virtual ~pacifier()
      {
      }
   // @}

   /*! \name Pacifier operation */
   std::string update(int complete, int total = 100);
   // @}
};

} // end namespace

#endif
