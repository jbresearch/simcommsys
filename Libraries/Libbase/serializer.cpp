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

#include "serializer.h"
#include <iostream>

namespace libbase {

using std::hex;
using std::dec;

// Determine debug level:
// 1 - Normal debug output only
// 2 - Trace calls
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// static variables

std::map<std::string, serializer::fptr>* serializer::cmap = NULL;
int serializer::count = 0;

// static functions

serializable* serializer::call(const std::string& base, const std::string& derived)
   {
   fptr func = (*cmap)[base + ":" + derived];
#if DEBUG>=2
   trace << "DEBUG (serializer): call(" << base+":"+derived << ") = " << (void *)func << "." << std::endl;
#endif
   if (func == NULL)
      return NULL;
   return (*func)();
   }

// constructor / destructor

serializer::serializer(const std::string& base, const std::string& derived,
      fptr func)
   {
   if (cmap == NULL)
      cmap = new std::map<std::string, fptr>;
#if DEBUG>=2
   trace << "DEBUG (serializer): new map entry [" << count << "] for (" << base+":"+derived << ") = " << (void *)func << "." << std::endl;
#endif
   (*cmap)[base + ":" + derived] = func;
   classname = derived;
   count++;
   }

serializer::~serializer()
   {
   count--;
   if (count == 0)
      delete cmap;
   }

} // end namespace
