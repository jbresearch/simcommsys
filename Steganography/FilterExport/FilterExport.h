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

#ifndef afx_filterexport_h
#define afx_filterexport_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include <fstream>


/////////////////////////////////////////////////////////////////////////////
// SFilterExportData
//

/*
  Data Version 1.00 (24 Apr 2002)
  initial version - last used filename.
*/
struct SFilterExportData {
   char     sPathName[256];
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterExportApp
// See FilterExport.cpp for the implementation of this class
//

/*
  Version 1.00 (24 Apr 2002)
  initial version

  Version 1.10 (5 Nov 2002)
  added scripting support; we also now tile the image row by row, keeping the same
  tile area as suggested (before we had to export the whole thing at once).

  Version 1.11 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.20 (13 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class CFilterExportApp : public CWinApp, public libwin::CPSPlugIn
{
protected:
   SFilterExportData* m_sData;
   std::ofstream file;

protected:
   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterExportApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterExportApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterExportApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif

