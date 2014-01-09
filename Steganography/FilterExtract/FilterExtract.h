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

#ifndef afx_filterextract_h
#define afx_filterextract_h

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
// SFilterExtractData
//

/*
  Data Version 1.00 (30 Nov 2001)
  copied from FilterEmbed - all entries unused.

  Data Version 1.01 (30 Nov 2001)
  deleted strength; kept pathname to allow comparison of extraction to source.

  Data Version 1.10 (15 Feb 2002)
  added embedding density.

  Data Version 1.20 (27 Mar 2002)
  added Codec and Puncture filenames.

  Data Version 1.21 (21 Apr 2002)
  added boolean to determine whether or not we are interleaving.
  also added preset strength value (to avoid estimation errors) and a boolean to
  indicate if we have this or not.

  Data Version 1.30 (7-9 May 2002)
  added source type field (for comparison); added seeds for embedding system,
  interleaver and random source; renamed Density to InterleaverDensity; renamed
  PathName to Source; renamed Strength to EmbedStrength; added bandwidth expansion
  rate as EmbedRate; added filenames for storing results (embedded & extracted
  sequences in signal space, uniform modulation sequence, decoded sequence).

  Data Version 1.31 (10 May 2002)
  added booleans to indicate that we want to compute the SNR and BER.

  Data Version 1.32 (6 Jun 2002)
  added filename for storing numerical results.

  Data Version 1.33 (25 Jul 2002)
  added feedback type.

  Data Version 1.34 (16 Nov 2002)
  reversed the order of BER and SNR flags; also renamed these PrintBER and PrintSNR,
  since the user does not really care what needs to be computed, but only what should
  be printed; added flags for printing estimate (usually this means the SNR estimate)
  and the chi square metric.

  Data Version 1.35 (19 Feb 2003)
  added entries for storing filenames in which to save embedded and extracted vectors
  in image domain.
*/
struct SFilterExtractData {
   // embedding system
   int      nEmbedSeed;
   int      nEmbedRate;
   double   dEmbedStrength;
   bool     bPresetStrength;
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
   // results storage
   char     sResults[256];
   char     sEmbeddedImage[256];
   char     sExtractedImage[256];
   char     sEmbedded[256];
   char     sExtracted[256];
   char     sUniform[256];
   char     sDecoded[256];
   // channel parameter computation / feedback
   bool     bPrintBER;
   bool     bPrintSNR;
   bool     bPrintEstimate;
   bool     bPrintChiSquare;
   int      nFeedback;
   };

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractApp
// See FilterExtract.cpp for the implementation of this class
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
  added support for pre-set strength value and also a function to compute embedding
  strength from the stego-signal power as used by Marvel.
  added error-checking of bits where no real data is embedded.
  fixed a bug in converting from a gaussian variate to a uniform one: the function
  should be y = 1/2 * (1 + erf(x/sqrt(2)); the sqrt(2) factor was missing.

  Version 1.05 (23 Apr 2002)
  removed userCanceledError at FilterFinish, because this was not allowing Photoshop
  to record this filter's use in the actions palette, etc.

  Version 1.06 (24 Apr 2002)
  added hook to save signal-space data.

  Version 1.07 (29 Apr 2002)
  added hook to save uniform random sequence; also disabled the question to save the
  extracted data unless a source file was supplied (for comparison).

  Version 1.10 (7-9 May 2002)
  revamped filter architecture; made filter operate in multi-tile mode (single-pass);
  added support for bandwidth expansion; added support for saving results (for further
  analysis in Matlab).

  Version 1.11 (10 May 2002)
  added check boxes to allow the user to activate/disable computing the BER and SNR
  for the channel; results are displayed at the end of the filter in an appropriate
  dialog box. Also fixed a few bugs in the result saving sequence and added another
  two parameters to EstimateSNR: i) a pointer to double is supplied, if not NULL, then
  the real SNR is returned there; ii) a vector with the actual embedded signal is also
  supplied to be able to compute the real SNR for all cases.
  Also, added GetOutputSize() and GetInputSize() functions to obtain the correct
  block size to use; note that in these functions we need to round the number of bits
  returned from codec to ensure that we use the required value (was getting different
  results in the debug and release builds before).

  Version 1.12 (11 May 2002)
  fixed a bug in the main dialog - the "clear extracted" command was not clearing the
  respective filename.

  Version 1.13 (4 Jun 2002)
  fixed a bug in computing the SNR: replaced -20*log10(8*dLambda*sqrt(dRate)) by
  -20*log10(dLambda*sqrt(2)*sqrt(dRate)).

  Version 1.14 (6 Jun 2002)
  added a facility for user to save computer SNR/BER to a text file.

  Version 1.15 (9 Jun 2002)
  as a temporary fix, modified FEC decoder to use real SNR instead of estimated (this
  should be replaced by user option, and the estimator should be improved).

  Version 1.20 (19,25 Jul 2002)
  included encoding/embedding functions within this class to allow representing the
  channel error back into image space. Also upgraded system to multi-pass architecture
  to allow this, as follows:
  * Iteration 1: extract message
  * End: de-interleave, decode message; compute SNR, BER, channel error (as needed)
  * Iteration 2: write back channel error
  * Finish: display SNR, BER if requested
  The type of feedback is chosen by the user: either nothing is written back into image
  space, or else any one of channel error, absolute channel error

  Version 1.21 (26 Jul 2002)
  added log probability ratio feedback, with normalization.

  Version 1.30 (6 Nov 2002)
  added scripting support.

  Version 1.31 (8 Nov 2002)
  modified PluginMain to utilize main function now found in PSPlugIn 1.52.

  Version 1.32 (14 Nov 2002)
  bug fixes: only do a second iteration if there is feedback; do not write preset
  strength to results file.

  Version 1.40 (15 Nov 2002)
  moved embedding/extraction routines to a new class CStegoSystem - this makes the code
  for the embed/extract filters leaner, and also simplifies the process of keeping them
  in sync.

  Version 1.41 (16 Nov 2002)
  * changed data structure - reversed the order of BER and SNR flags; also renamed these
  PrintBER and PrintSNR, since the user does not really care what needs to be computed,
  but only what should be printed; added flags for printing estimate (usually this means
  the SNR estimate) and the chi square metric.
  * modified the user interface - added flag entry for estimate and chi square; also
  changed the order of flags to reflect the order they're printed; changed the output
  dialog to reflect the new order; changed the file output routine to reflect the new
  order.
  * Chi square metric output still not functional.

  Version 1.42 (16 Nov 2002)
  added output of chi square metric.

  Version 1.43 (18 Nov 2002)
  modified printing of chi square metric to reflect the change introduced in StegoSystem
  1.02, where the returned value is the computed chi square, rather than its associated
  probability value.

  Version 1.44 (19 Feb 2003)
  * modified user interface - added entries for saving embedded and extracted vectors in
  image domain.
  * modified data structure to include above entries.
  * modified filter to save the above vectors.

  Version 1.50 (13 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.51 (1 Dec 2006)
  * updated to accomodate changes to stegosystem (move from libwin to libcomm and change of name).
*/

class CFilterExtractApp : public CWinApp, public libwin::CPSPlugIn, protected libcomm::stegosystem
{
protected:
   SFilterExtractData* m_sData;
        int               m_nIteration;
   double            m_dBER, m_dSNRreal, m_dSNRest, m_dChiSquare;
   int               m_nCount, m_nLength;
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
   CFilterExtractApp();

   void FilterAbout(void);
   void FilterStart(void);
   void FilterContinue(void);
   void FilterFinish(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterExtractApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CFilterExtractApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif

