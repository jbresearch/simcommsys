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

#ifndef afx_filterorphans_h
#define afx_filterorphans_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "matrix.h"
#include "timer.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterOrphansData
//

/*
  Data Version 1.00 (7 Apr 2002)
  initial version (KeepNoise).

  Data Version 1.01 (12 Apr 2002)
  added nWeight = minimum number of neighbours that must be on in order to keep
  current pixel.
*/
struct SFilterOrphansData {
   bool  bKeepNoise;
   int   nWeight;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansApp
// See FilterOrphans.cpp for the implementation of this class
//

/*
  Version 1.00 (7 Apr 2002)
  initial version - for some reason this won't work in bitmap mode - PhotoShop complains
  that the filter cannot work with single-channel images.

  Version 1.10 (12 Apr 2002)
  modified dialog to what it should have been - user has two options: to keep the noise
  insead of cancelling it (which is the default) and to choose the minimum number of
  neighbours that must be on in order to keep the current pixel.

  Version 1.11 (23 Apr 2002)
  modified PiPL so that filter is only active in grayscale mode. Also modified algorithm
  so that it thresholds image at 0.5, and output pixels are set to black or white as
  appropriate.

  Version 1.20 (6 Nov 2002)
  added scripting support.

  Version 1.21 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.30 (13 Nov 2006)
  * updated to use library namespaces.
*/

class CFilterOrphansApp : public CWinApp, public libwin::CPSPlugIn
{
protected:
   SFilterOrphansData* m_sData;

protected:
   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterOrphansApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterOrphansApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterOrphansApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
