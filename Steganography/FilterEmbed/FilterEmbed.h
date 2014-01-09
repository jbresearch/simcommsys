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

#ifndef afx_filterembed_h
#define afx_filterembed_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSPlugIn.h"
#include "stegosystem.h"
#include "matrix.h"
#include "vector.h"
#include "timer.h"


/////////////////////////////////////////////////////////////////////////////
// SFilterEmbedData
//

/*
  Data Version 1.10 (15 Feb 2002)
  added embedding density.

  Data Version 1.20 (27 Mar 2002)
  added Codec and Puncture filenames.

  Data Version 1.21 (14 Apr 2002)
  made embedding strength a double instead of int

  Data Version 1.22 (21 Apr 2002)
  added boolean to determine whether or not we are interleaving.

  Data Version 1.30 (25 Apr 2002)
  added seeds for embedding system, interleaver and random source;
  added source type field; renamed Density to InterleaverDensity;
  renamed PathName to Source; renamed Strength to EmbedStrength

  Data Version 1.31 (26 Apr 2002)
  added bandwidth expansion rate as EmbedRate.
*/
struct SFilterEmbedData {
   // embedding system
   int      nEmbedSeed;
   int      nEmbedRate;
   double   dEmbedStrength;
   // channel interleaver
   bool     bInterleave;
   int      nInterleaverSeed;
   double   dInterleaverDensity;
   // source data
   int      nSourceType;
   int      nSourceSeed;
   char     sSource[256];
   // codec and puncture pattern
   char     sCodec[256];
   char     sPuncture[256];
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedApp
// See FilterEmbed.cpp for the implementation of this class
//

/*
  Version 1.00 (undated)
  initial version

  Version 1.01 (6 Apr 2002)
  added DisplayProgress in FilterContinue, since automatic progress update was removed
  in PSPlugIn 1.41. Also modified the existing DisplayProgress calls to use the new
  multi-pass support, for a more meaningful display.

  Version 1.02 (7 Apr 2002)
  modified the PiPL file to flag which modes are supported.

  Version 1.03 (15 Apr 2002)
  fixed a bug in the user interface by adding "clear" buttons for the source file,
  codec, and puncturing system.

  Version 1.04 (21 Apr 2002)
  added selector to disable/enable variable-density interleaving
  added function to compute embedding strength from the stego-signal power as used
  by Marvel.
  added support for NULL embedding (ie. sequence embedded is all-zero if no filename
  is passed along as input data).
  fixed a bug in converting from a uniform variate to a gaussian one: the function
  should be y = erfinv(2*x-1) * sqrt(2); the sqrt(2) factor was missing.
  added information in dialog box about raw capacity, usable capacity, code size,
  and data rate.

  Version 1.10 (25 Apr 2002)
  revamped filter architecture.

  Version 1.11 (26 Apr 2002)
  made filter operate in multi-tile mode (single-pass); added bandwidth expansion
  capability.

  Version 1.12 (8 May 2002)
  modified tile size selector to go for a minimum of one row (to ensure that selected
  tile size is not empty). Also fixed memory leakage by ensuring that the codec and
  puncture pattern are deleted in FilterFinish.

  Version 1.13 (10 May 2002)
  added GetOutputSize() and GetInputSize() functions to obtain the correct block size
  to use; note that in these functions we need to round the number of bits returned
  from codec to ensure that we use the required value (was getting different results
  in the debug and release builds before).

  Version 1.20 (6 Nov 2002)
  added scripting support.

  Version 1.21 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.30 (15 Nov 2002)
  moved embedding/extraction routines to a new class CStegoSystem - this makes the code
  for the embed/extract filters leaner, and also simplifies the process of keeping them
  in sync.

  Version 1.40 (13 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.41 (1 Dec 2006)
  * updated to accomodate changes to stegosystem (move from libwin to libcomm and change of name).
*/

class CFilterEmbedApp : public CWinApp, public libwin::CPSPlugIn, protected libcomm::stegosystem
{
protected:
   SFilterEmbedData* m_sData;
   libbase::vector<double>    m_vdMessage;

protected:
   // StegoSystem overrides
   int GetImagePixels() const { return GetImageWidth() * GetImageHeight() * GetPlanes(); };
   void DisplayProgress(const int nComplete, const int nTotal, const int nIteration, const int nTotalIterations) const { DisplayTotalProgress(nComplete, nTotal, nIteration, nTotalIterations); };

   // PSPlugIn overrides
   void ShowDialog(void);
   void InitPointer(char* sData);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIWriteDescriptor token);
   void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags);

public:
   CFilterEmbedApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterEmbedApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterEmbedApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif


