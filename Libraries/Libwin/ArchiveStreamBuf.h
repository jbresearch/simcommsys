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

#ifndef __archivestreambuf_h
#define __archivestreambuf_h

#include <iostream>
#include <ios>

/*
   \version 1.10 (6 Nov 2006)
   - defined class and associated data within "libwin" namespace.
   - removed pragma once directive, as this is unnecessary
   - changed unique define to conform with that used in other libraries
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libwin {

class CArchiveStreamBuf : public std::streambuf
{
protected:
   CArchive *ar;
public:
        CArchiveStreamBuf(CArchive* ar);
   virtual ~CArchiveStreamBuf() {};
   int underflow();
   int overflow(int nCh = EOF);
};

inline int CArchiveStreamBuf::underflow()
   {
   char c;
   return ar->Read(&c, 1) == 1 ? c : EOF;
   }

inline int CArchiveStreamBuf::overflow(int nCh)
   {
   char c = nCh;
   ar->Write(&c, 1);
   return 1;
   };

} // end namespace

#endif
