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

#ifndef __mboxstreambuf_h
#define __mboxstreambuf_h

#include <iostream>
#include <ios>

/*
   \version 1.10 (29 Apr 2002)
  changed the meaning of \r: when a CR is received, it is translated into a newline
  _within_ the current mbox. A newline means as before (ie. output buffer into mbox)

   \version 1.20 (6 Nov 2006)
   - defined class and associated data within "libwin" namespace.
   - removed pragma once directive, as this is unnecessary
   - changed unique define to conform with that used in other libraries
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libwin {

   class CMBoxStreamBuf : public std::streambuf
{
protected:
   CString buffer;
public:
        CMBoxStreamBuf();
        virtual ~CMBoxStreamBuf();
   int underflow();
   int overflow(int nCh = EOF);
};

inline int CMBoxStreamBuf::underflow()
   {
   return EOF;
   }

inline int CMBoxStreamBuf::overflow(int nCh)
   {
   if(nCh=='\n')
      {
      if(!buffer.IsEmpty())
         AfxMessageBox(buffer);
      buffer = "";
      }
   else if(nCh=='\r')
      buffer += '\n';
   else
      buffer += char(nCh);
   return 1;
   };

} // end namespace

#endif
