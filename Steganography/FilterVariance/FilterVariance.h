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

#ifndef afx_filtervariance_h
#define afx_filtervariance_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "matrix.h"
#include "timer.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterVarianceData
//

/*
  Data Version 1.00 (30 Nov 2001)
  keeps radius for which to work local variance.

  Data Version 1.01 (30 Nov 2001)
  added a scale entry.
*/
struct SFilterVarianceData {
   int   nRadius;
   int   nScale;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceApp
// See FilterVariance.cpp for the implementation of this class
//

/*
  Version 1.00 (undated)
  initial version

  Version 1.10 (5 Apr 2002)
  made the dialog neater by adding a region size info box (auto-updating with the
  radius edit box); also added auto-scaling facility. In the data block, this is
  indicated by a scale of zero (would never want that value anyway); in the dialog
  however, I added a check box, which automatically disables the scale edit box.
  Also changed the filter to work on two passes if we want auto-scaling (this is
  necessary or else we get different scales for each tile). Support for multiple
  iterations has just been added to PSPlugIn 1.40.

  Version 1.11 (6 Apr 2002)
  added DisplayProgress in FilterContinue, since automatic progress update was removed
  in PSPlugIn 1.41.

  Version 1.12 (7 Apr 2002)
  modified the PiPL file to flag which modes are supported.

  Version 1.20 (6 Nov 2002)
  added scripting support.

  Version 1.21 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.30 (13 Nov 2006)
  * updated to use library namespaces.
*/

class CFilterVarianceApp : public CWinApp, public libwin::CPSPlugIn
{
protected:
   SFilterVarianceData* m_sData;
        int m_nIteration;
   double m_dScale;

protected:
   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterVarianceApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterVarianceApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterVarianceApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif

