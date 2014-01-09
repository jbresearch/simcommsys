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

#ifndef __imagefile_h
#define __imagefile_h

#include "config.h"

/*******************************************************************************

 Version 1.00 (7 Dec 2006)
 * initial version
 * class meant to encapsulate the data and functions dealing with image files,
 potentially containing several images.
 * this is effectively a container for images, together with the functions
 necessary to save and load to files or other streams.

 *******************************************************************************/

namespace libimage {

class imagefile {
public:
   // Construction / destruction
   imagefile();
   ~imagefile();

   // File format

   // Saving / loading

};

} // end namespace

#endif
