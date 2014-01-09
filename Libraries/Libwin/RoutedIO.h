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

#ifndef __routedio_h
#define __routedio_h

#include "MBoxStreamBuf.h"
#include "TraceStreamBuf.h"

/*
   \version 1.00 (10 Nov 2006)
   - Initial version - this class encapsulates the routing of standard I/O streams to
  the debug trace (for clog/cout) and to a message box (for cerr). This makes any
  derivative of this class automatically take over these facilities.
*/

namespace libwin {

class CRoutedIO
   {
   private:
      CMBoxStreamBuf    m_msgbox;
      CTraceStreamBuf   m_tracer;
   public:
      CRoutedIO();
   };

} // end namespace

#endif
