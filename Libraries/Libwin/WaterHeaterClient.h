#ifndef __waterheaterclient_h
#define __waterheaterclient_h

#include "StdAfx.h"

/*
  Version 1.00 (25 Dec 2006)
  * Initial version, uncompleted
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

}; // end namespace

#endif
