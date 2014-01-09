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

#ifndef afx_automategraphing_h
#define afx_automategraphing_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSAutomate.h"

/////////////////////////////////////////////////////////////////////////////
// SAutomateGraphingData
//

/*
  Data Version 1.00 (7 Nov 2002)
  initial version - contains variables required for ATM filter

  Data Version 1.10 (12 Nov 2002)
  added files with system parameters (input) and results (output); range of embedding
  strengths; range of JPEG compression quality (with flag to indicate if requested).

  Data Version 1.11 (15 Nov 2002)
  added flags for requested outputs (currently BER, SNR, ChiSquare); added flag for
  allowing/disabling preset strength.

  Data Version 1.12 (16 Nov 2002)
  renamed output flags PrintBER, PrintSNR, and PrintChiSquare since the user does not
  really care what needs to be computed, but only what should be printed; also added
  flag for printing estimate (usually this means the SNR estimate).
*/
struct SAutomateGraphingData {
   // files with system parameters (input) and results (output)
   char     sParameters[256];
   char     sResults[256];
   // system options
   bool     bJpeg;
   bool     bPresetStrength;
   // variables - range of embedding strengths
   double   dStrengthMin;
   double   dStrengthMax;
   double   dStrengthStep;
   // variables - range of JPEG compression quality (if requested)
   int      nJpegMin;
   int      nJpegMax;
   int      nJpegStep;
   // requested outputs
   bool     bPrintBER;
   bool     bPrintSNR;
   bool     bPrintEstimate;
   bool     bPrintChiSquare;
   };

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp
// See AutomateGraphing.cpp for the implementation of this class
//

/*
  Version 1.00 (7-8 Nov 2002)
  * initial version
  * there is a bug (not necessarily in this class), that was detected and still unsolved:
  currently this plugin merely collects the data necessary for an ATM filter, and then
  plays back the atm filter event. On final exit of this class, there is a memory leakage
  of two blocks (33 and 40 bytes) which is still unsolved.

  Version 1.10 (11-12 Nov 2002)
  * modified interface to allow entry of: filenames for system parameters and results,
  and settings for variables.
  * added scripting support, in conformance with PSAutomate 1.10.
  * changed the pData parameter in PluginMain from const void* to void*, in conformance
  with PSAutomate 1.10.

  Version 1.11 (14 Nov 2002)
  * wrote Process routine for version 1.10.

  Version 1.12 (15 Nov 2002)
  modified interface to allow choice of requested outputs (currently BER, SNR, ChiSquare)
  and setting for preset strength; integrated the use of these settings in Process, except
  for ChiSquare, since the current extract filter does not yet do this.

  Version 1.13 (16 Nov 2002)
  * changed data structure - renamed output flags PrintBER and PrintSNR, since the user
  does not really care what needs to be computed, but only what should be printed; added
  flag for printing estimate (usually this means the SNR estimate).
  * modified the user interface - added flag entry for estimate.

  Version 1.14 (18 Nov 2002)
  modified dialog so that browsing for the results and parameters file defaults to the
  current filename/directory, if present.

  Version 1.20 (13 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class CAutomateGraphingApp : public CWinApp, public libwin::CPSAutomate
{
protected:
   SAutomateGraphingData* m_sData;
   struct SParameters {
      int nFilterType;
      union UFilterSettings {
         struct SFilterATM {
            int      nRadius;
            int      nAlpha;
            } sFilterATM;
         struct SFilterAW {
            int      nRadius;
            double   dNoise;
            } sFilterAW;
         struct SFilterWavelet {
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
            } sFilterWavelet;
         } uFilterSettings;
      } m_sParameters;

protected:
   // internal functions
   void ReadParameters();
   void WriteHeader();
   void DoExtract(double dStrength);

   // virtual overrides - data handling
   void ShowDialog(void);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIActionDescriptor descriptor);
   void ReadScriptParameters(PIActionDescriptor descriptor);

public:
   CAutomateGraphingApp();
   virtual ~CAutomateGraphingApp();

   // virtual overrides - plug-in interface
   void About(void);
   void Process(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAutomateGraphingApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CAutomateGraphingApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
