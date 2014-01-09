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

#ifndef afx_filterenergy_h
#define afx_filterenergy_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "rvstatistics.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterEnergyData
//

/*
  Data Version 1.00 (23 Apr 2002)
  initial version - scale, unused.

  Data Version 1.01 (2 Nov 2002)
  removed scale (since we don't need it anyway); added filename where we shall
  write the computed result (if empty we assume the user wants to see the result
  on-screen); added boolean to choose whether the result should be appended to the
  file; added booleans to determine what should be computed.
*/
struct SFilterEnergyData {
   char  sFileName[256];
   bool  bAppend;
   bool  bDisplayVariance;
   bool  bDisplayEnergy;
   bool  bDisplayPixelCount;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyApp
// See FilterEnergy.cpp for the implementation of this class
//

/*
  Version 1.00 (23 Apr 2002)
  initial version

  Version 1.10 (2 Nov 2002)
  revamped user interface: we now allow the user to select between displaying results
  on-screen or appending to a file; added scripting support.

  Version 1.11 (2 Nov 2002)
  * modified scripting support technique: removed scripting.h, integrating the necessary
  elements in the Adobe SDK resources file and moving the key definitions to a new
  header file (scriptingkeys.h) in LibWin; also, we now use different SuiteID and EventID
  for the debug and release builds, and also use the full plugin and vendor names within
  the dictionary (scripting) resource. Together, these steps clearly identify which
  build version of the filter was recorded.
  * also, modified the read/write parameters routines to use the new functions provided
  in PSPlugIn 1.51.

  Version 1.12 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.20 (13 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class CFilterEnergyApp : public CWinApp, public libwin::CPSPlugIn
{
protected:
   SFilterEnergyData* m_sData;
   libbase::rvstatistics   rv;

protected:
   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterEnergyApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterEnergyApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterEnergyApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
