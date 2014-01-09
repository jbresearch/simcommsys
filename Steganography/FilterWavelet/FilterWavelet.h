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

#ifndef afx_filterwavelet_h
#define afx_filterwavelet_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "matrix.h"
#include "timer.h"
#include "waveletfilter.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterWaveletData
//

/*
  Data Version 1.01 (28 Nov 2001)
  added variable bKeepNoise to indicate if we want the filter to return the noise
  component rather than the filtered image.

  Data Version 1.10 (2 Dec 2001)
  added tile width and height sizes.

  Data Version 1.11 (21 Apr 2002)
  added thresholding type (ie. hard and soft)

  Data Version 1.12 (29 Apr 2002)
  renamed data members and organized them better; added threshold selector member;
  added boolean to indicate whole-image tiling (useful for multi-file processing).

  Data Version 1.13 (30 Apr 2002)
  added wavelet parameter variable (used to determine the length / number of vanishing
  moments, etc. for the specified wavelet type).
*/
struct SFilterWaveletData {
   // wavelet basis
   int      nWaveletType;
   int      nWaveletPar;
   int      nWaveletLevel;
   // thresholding
   int      nThreshType;
   int      nThreshSelector;
   double   dThreshCutoff;
   // tiling
   int      nTileX;
   int      nTileY;
   bool     bWholeImage;
   // other
   bool     bKeepNoise;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletApp
// See FilterWavelet.cpp for the implementation of this class
//

/*
  Version 1.00 (undated)
  initial version

  Version 1.01 (6 Apr 2002)
  added DisplayProgress in FilterContinue, since automatic progress update was removed
  in PSPlugIn 1.41.

  Version 1.02 (7 Apr 2002)
  modified the PiPL file to flag which modes are supported.

  Version 1.03 (12 Apr 2002)
  modified dialog to include a check box to apply filter to whole image at once, rather
  than the suggested tile size.

  Version 1.04 (14 Apr 2002)
  modified dialog box display function so that "whole image" is checked even if last
  tile size was _exactly_ equal to the image size.

  Version 1.10 (15-16 Apr 2002)
  updated plugin to use the revamped waveletfilter module, in a dual-pass scheme.

  Version 1.11 (21 Apr 2002)
  added choice between hard and soft thresholding.

  Version 1.20 (29 Apr 2002)
  cleaned up user interface & renamed data members; made waveletfilter a private base
  of class; added visu threshold selection and paved the way for other threshold
  selectors; added in-filter progress display.

  Version 1.30 (30 Apr 2002)
  added support for Haar, Beylkin, Coiflet, Daubechies, Symmlet, Vaidyanathan, and
  Battle-Lemarie wavelets with a number of parameters for each, as in wavelet 1.30.

  Version 1.31 (17 Oct 2002)
  fixed a minor bug (performance enhancement only) - limiter process is now only done
  twice if KeepNoise was true.

  Version 1.40 (6 Nov 2002)
  added scripting support; also changed FilterStart to check bWholeImage before setting
  the tile width (before this was only done in ShowDialog, which meant [bug] that if
  the user ran the filter again with WholeImage on a new image without ShowDialog, then
  the old tile size would be used). Also, modified FilterStart to first call the base
  class's FilterStart, then change the tile size, and then call IterationStart again
  to set the new tile sizes.

  Version 1.41 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.50 (13 Nov 2006)
  * updated to use library namespaces.
*/

class CFilterWaveletApp : public CWinApp, public libwin::CPSPlugIn, private libimage::waveletfilter
{
protected:
   SFilterWaveletData* m_sData;
        int m_nIteration;

protected:
   // filter overrides
   void display_progress(const int done, const int total) const { DisplayTileProgress(done, total, m_nIteration, 2); };

   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterWaveletApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterWaveletApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterWaveletApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
