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

#ifndef afx_filterlevels_h
#define afx_filterlevels_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "matrix.h"
#include "timer.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterLevelsData
//

/*
  Data Version 1.00 (5 Jan 2005)
  initial version (no data).
*/
struct SFilterLevelsData {
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsApp
// See FilterLevels.cpp for the implementation of this class
//

/*
  Version 1.00 (5 Jan 2005)
  initial version - this plugin is meant to automatically determine the best black and
  white points for a grayscale scan of lineart or text. These are determined based on
  how the chosen thresholds affect the neighborhood statistics (this makes it similar
  to what is manually done when holding Alt while dragging the white/black points on
  Photoshop; the ideal black point is where the text areas start to link well together,
  while the ideal white point is where the background areas start to become clean).
  The algorithms follow my Matlab implementation for the 2003-2004 APTs.

  Version 1.10 (13 Nov 2006)
  * updated to use library namespaces.
*/

class CFilterLevelsApp : public CWinApp, public libwin::CPSPlugIn
{
// constants
protected:
   static const int  m_nLevels;

protected:
   SFilterLevelsData* m_sData;
        int               m_nIteration;
   double            m_dWhite, m_dBlack;
   libbase::vector<int>       m_viHistogram;
   libbase::matrix<int>       m_miNeighbors;

protected:
   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterLevelsApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterLevelsApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterLevelsApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
