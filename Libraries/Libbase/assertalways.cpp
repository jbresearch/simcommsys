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

/*!
 * \file
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "assertalways.h"

#include <iostream>
#include <string>
#include <cstdlib>

namespace libbase {

// Debugging tools

void reportassertionandfail(const char *expression, const char *file, int line)
   {
   std::string s;
   s = "assertion " + std::string(expression) + " failed.";
   reporterrorandfail(s.c_str(), file, line);
   }

void reporterrorandfail(const char *expression, const char *file, int line)
   {
   std::cerr << "ERROR (" << file << " line " << line << "): " << expression
         << std::endl;
   exit(1);
   }

} // end namespace
