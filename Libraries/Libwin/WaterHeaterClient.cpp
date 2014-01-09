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

#include "stdafx.h"
#include "WaterHeaterClient.h"
#include "config.h"
#include <iostream>

namespace libwin {

using std::cerr;
using libbase::trace;

// Internal helper functions

double CWaterHeaterClient::KTemp(double V, double Vcjc) const
   {
   // This function returns the Temperature in C, given the measured TC
   // voltage and the CJC voltage. For K type thermocouple. Valid for 0-500C.
   // Ref: http://srdata.nist.gov/its90/type_k/kcoefficients_inverse_html
   // Matlab original: Mario Farrugia, 2005 (version C)
   // Conversion to C: Johann A. Briffa, 2006

   // Coefficients are for voltage in uV
   const double c[] = { 0, 0.02508355, 7.860106E-8, -2.503131E-10, 8.31527E-14,
      -1.228034E-17, 9.804036E-22, -4.41303E-26, 1.057734E-30, -1.052755E-35 };

   V = (V+Vcjc)*1E6;
   double v = 1;
   double t = c[0];
   for(int i=1; i<10; i++)
      {
      v *= V;
      t += c[i] * v;
      }

   return t;
   }

double CWaterHeaterClient::KVolt(double T) const
   {
   // This function returns the K type Voltage in Volts given the temperature in C
   // Matlab original: Mario Farrugia, 2005
   // Conversion to C: Johann A. Briffa, 2006

   // Coefficients are for voltage in uV
   const double a[] = { -0.0001183432, 118.5976, 126.9686 };
   const double c[] = { -17.600413686, 38.921204975, 0.018558770032, -0.000099457592874,
      3.1840945719E-07, -5.6072844889E-10, 5.6075059059E-13, -3.2020720003E-16,
      9.7151147152E-20, -1.2104721275E-23 };

   double v = T - a[2];
   v = a[1] * v*v;
   v = a[0] * exp(v);

   double t = 1;
   v += c[0];
   for(int i=1; i<10; i++)
      {
      t *= T;
      v += c[i] * t;
      }

   return v*1E-6;
   }

// Constructor/destructor

CWaterHeaterClient::CWaterHeaterClient(CString sPort)
   {
   trace << "Opening port (" << sPort << ").\n";
   m_hComm = CreateFile(sPort, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
   if(m_hComm == INVALID_HANDLE_VALUE)
      {
      cerr << "Port access failed (" << sPort << ").\n";
      exit(1);
      }

   COMMTIMEOUTS CommTimeouts;
   CommTimeouts.ReadIntervalTimeout = 500;
   CommTimeouts.ReadTotalTimeoutMultiplier = 0;
   CommTimeouts.ReadTotalTimeoutConstant = 1000;
   CommTimeouts.WriteTotalTimeoutMultiplier = 0;
   CommTimeouts.WriteTotalTimeoutConstant = 1000;
   SetCommTimeouts(m_hComm, &CommTimeouts);

   DCB dcb = {0};
   dcb.DCBlength = sizeof(dcb);
   if(!BuildCommDCB("9600,n,8,1", &dcb))
      {
      cerr << "Port settings failed.\n";
      exit(1);
      }

   }

CWaterHeaterClient::~CWaterHeaterClient()
   {
   trace << "Closing serial port.\n";
   CloseHandle(m_hComm);
   }

// User functions

   //{
   //const int nBufLen = 1024;
   //char pBuf[nBufLen+1];
   //
   //DWORD dwBytes;
   //WriteFile(m_hComm, pBuf, nBytes, &dwBytes, NULL);
   //ReadFile(m_hComm, pBuf, nBufLen, &dwBytes, NULL);
   //}

} // end namespace
