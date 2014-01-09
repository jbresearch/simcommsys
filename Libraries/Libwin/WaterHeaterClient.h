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

#ifndef __waterheaterclient_h
#define __waterheaterclient_h

#include "StdAfx.h"

/*
   \version 1.00 (25 Dec 2006)
   - Initial version, uncompleted
*/

namespace libwin {

class CWaterHeaterClient
{
private:
   HANDLE   m_hComm;
private:
   double KTemp(double V, double Vcjc) const;
   double KVolt(double T) const;
public:
   // Constructor/destructor
        CWaterHeaterClient(CString sPort);
        virtual ~CWaterHeaterClient();
   // User functions
};

} // end namespace

#endif
