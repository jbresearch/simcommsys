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

#ifndef afx_filteratm_h
#define afx_filteratm_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "matrix.h"
#include "timer.h"
#include "atmfilter.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterATMData
//

/*
  Data Version 1.01 (28 Nov 2001)
  added variable bKeepNoise to indicate if we want the filter to return the noise
  component rather than the filtered image.
*/
struct SFilterATMData {
   int   nAlpha;
   int   nRadius;
   bool  bKeepNoise;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterATMApp
// See FilterATM.cpp for the implementation of this class
//

/*
  Version 1.00 (undated)
  initial version

  Version 1.01 (6 Apr 2002)
  added DisplayProgress in FilterContinue, since automatic progress update was removed
  in PSPlugIn 1.41.

  Version 1.10 (6 Apr 2002)
  improved user interface by adding a region size info box (auto-updating with the
  radius edit box); also added a slider that auto-updates in sync with the alpha edit
  box and radius edit box. The range depends on radius and the position on alpha.

  Version 1.11 (7 Apr 2002)
  modified the PiPL file to flag which modes are supported.

  Version 1.12 (7 Apr 2002)
  set the default value of KeepNoise to false (was undefined).

  Version 1.20 (1 Nov 2002)
  updated plugin to use the filter-derived atmfilter module, as a private base; added
  in-filter progress display; added scripting support.

  Version 1.21 (2 Nov 2002)
  * modified scripting support technique: removed scripting.h, integrating the necessary
  elements in the Adobe SDK resources file and moving the key definitions to a new
  header file (scriptingkeys.h) in LibWin; also, we now use different SuiteID and EventID
  for the debug and release builds, and also use the full plugin and vendor names within
  the dictionary (scripting) resource. Together, these steps clearly identify which
  build version of the filter was recorded.
  * also, modified the read/write parameters routines to use the new functions provided
  in PSPlugIn 1.51.

  Version 1.22 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.30 (13 Nov 2006)
  * updated to use library namespaces.
*/

class CFilterATMApp : public CWinApp, public libwin::CPSPlugIn, private libimage::atmfilter<double>
{
protected:
   SFilterATMData* m_sData;

protected:
   // filter overrides
   void display_progress(const int done, const int total) const { DisplayTileProgress(done, total); };

   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterATMApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterATMApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterATMApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
