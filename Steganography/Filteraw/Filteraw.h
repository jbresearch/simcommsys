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

#ifndef afx_filteraw_h
#define afx_filteraw_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "matrix.h"
#include "timer.h"
#include "awfilter.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterAWData
//

/*
  Data Version 1.00 (4 Apr 2002)
  Original version - contained radius, noise estimate, and a variable bKeepNoise to indicate
  if we want the filter to return the noise component rather than the filtered image.

  Data Version 1.10 (3 Mar 2003)
  added boolean to indicate whether or not the noise threshold should be automatically
  estimated. This was necessary to allow feedback of the actual noise threshold used in such
  cases into the data structure, while letting a "repeat filter" command to again estimate
  the noise value.
*/
struct SFilterAWData {
   int      nRadius;
   double   dNoise;
   bool     bAuto;
   bool     bKeepNoise;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterAWApp
// See FilterAW.cpp for the implementation of this class
//

/*
  Version 1.00 (undated)
  initial version

  Version 1.01 (6 Apr 2002)
  added DisplayProgress in FilterContinue, since automatic progress update was removed
  in PSPlugIn 1.41.

  Version 1.10 (6 Apr 2002)
  made the dialog neater by adding a region size info box (auto-updating with the
  radius edit box); also added auto noise-estimation facility. In the data block, this
  is indicated by a noise estimate of zero (would never want that value anyway); in the
  dialog however, I added a check box, which automatically disables the noise estimate
  edit box. Also changed the filter to work on two passes if we want auto estimation
  (this is necessary or else we get different estimates for each tile). Support for
  multiple iterations was added to PSPlugIn 1.40.

  Version 1.11 (7 Apr 2002)
  modified the PiPL file to flag which modes are supported.

  Version 1.12 (14 Apr 2002)
  set the default value of KeepNoise to false (was undefined).

  Version 1.20 (17 Oct 2002)
  updated plugin to use the filter-derived awfilter module, in a dual-pass scheme;
  made awfilter a private base of class; added in-filter progress display.

  Version 1.30 (31 Oct - 1 Nov 2002)
  added scripting support

  Version 1.31 (2 Nov 2002)
  * modified scripting support technique: removed scripting.h, integrating the necessary
  elements in the Adobe SDK resources file and moving the key definitions to a new
  header file (scriptingkeys.h) in LibWin; also, we now use different SuiteID and EventID
  for the debug and release builds, and also use the full plugin and vendor names within
  the dictionary (scripting) resource. Together, these steps clearly identify which
  build version of the filter was recorded.
  * also, modified the read/write parameters routines to use the new functions provided
  in PSPlugIn 1.51.

  Version 1.32 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.33 (3 Mar 2003)
  modified data structure so that if the noise threshold was automatically estimated, the
  actual value used is returned in the data structure.

  Version 1.40 (13 Nov 2006)
  * updated to use library namespaces.
*/

class CFilterAWApp : public CWinApp, public libwin::CPSPlugIn, private libimage::awfilter<double>
{
protected:
   SFilterAWData* m_sData;
        int m_nIteration;

protected:
   // filter overrides
   void display_progress(const int done, const int total) const { DisplayTileProgress(done, total, m_nIteration,  m_sData->dNoise==0 ? 2 : 1); };

   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterAWApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterAWApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterAWApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
