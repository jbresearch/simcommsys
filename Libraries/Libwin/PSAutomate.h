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

#ifndef __psautomate_h
#define __psautomate_h

#include "RoutedIO.h"
#include "timer.h"

#ifdef ADOBESDK

#include "PIFilter.h"
#include "PIUIHooksSuite.h"
#include "PIHandleSuite.h"

// Create a definition for exported functions
#ifndef DLLExport
#   define DLLExport extern "C" __declspec(dllexport)
#endif

/*
   \version 1.00 (8 Nov 2002)
  initial version; based initially on PSPlugIn 1.52.

   \version 1.01 (9 Nov 2002)
  finished off writing the high-level event playback functions for the streganography
  and binary suites.

   \version 1.02 (11 Nov 2002)
  bug fix: moved memory allocation for data block to a new private function, which is
  now called from Reload; added version checking for data block; removed de-allocation
  from Unload. [interim version - never used]

   \version 1.10 (12 Nov 2002)
   - based on Don Ashe's suggestions, and also following the Listener plugin as a sample,
  parameters are now only "stored" between calls through the scripting system. Thus, all
  data-block related code has been removed, and high-level routines for handling the
  scripting system have been added (similar to PSPlugIn). Derived classes are now
  responsible for creating and destroying their own data structures; this can be done
  in at least three ways: 1) by having the data variables members of the derived class,
  either separately or as a struct, 2) by allocating/deallocating memory in the class
  constructor/destructor respectively, or 3) by allocating/deallocating memory in the
  Reload/Unload calls, as specified by the photoshop API. It seems to me that option 2
  allows for the greatest flexibility with minimal overhead.
   - added a new virtual function Execute, which is called when the Photoshop DoIt
  message is received. This encapsulates all that needs to be done at that stage,
  including reading/writing script parameters, showing the dialog, and calling Process.
   - changed the pData parameter in PluginMain from const void* to void*

   \version 1.11 (14 Nov 2002)
   - added Photoshop event playback for the following events: Open, Close, ConvertMode
  SelectState, SaveJPEG, Revert.
   - added support for handling alias and reference values (had only descriptors before).
   - added acquire/release of Handle Suite (needed for handling aliases)

   \version 1.12 (16 Nov 2002)
  modified PlayeventExtract to conform with the changes in FilterExtract 1.41.

   \version 1.13 (19 Feb 2003)
  removed inclusion of PIDefines, which is part of the SDK sample code, and not the API
  itself.

   \version 1.14 (13 Oct 2003)
   - added definition for DLLExport - this was previously being taken from PIDefines, which
  is part of the SDK sample code and not of the API itself. Compilations of automation
  classes were broken since v1.13 when PIDefines was removed; somehow this has not been
  detected earlier.
   - added PlayeventSaveLZW for TIFF/LZW compressed save.

   \version 1.20 (6 Nov 2006)
   - defined class and associated data within "libwin" namespace.
   - removed pragma once directive, as this is unnecessary
   - changed unique define to conform with that used in other libraries
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.21 (10 Nov 2006)
   - made class a derivative of CRoutedIO.

   \version 1.22 (7 Nov 2007)
   - moved Adobe SDK includes here from stdafx.h.
   - made this module compile only when ADOBESDK is defined.
*/

namespace libwin {

class CPSAutomate : public CRoutedIO
{
private:
   libbase::timer m_tDuration;      // operation timer

   // Photoshop data & basic suite
   void*          m_pData;
   SPBasicSuite*  m_pBasic;

protected:
   // Photoshop suites (accessed by derived classes)
   PSActionDescriptorProcs*   m_pActionDescriptor;
   PSActionReferenceProcs*    m_pActionReference;
   PSActionControlProcs*      m_pActionControl;
   PSActionListProcs*         m_pActionList;
   PSHandleSuite1*            m_pHandle;

private:
   // internal utility functions
   CString KeyToString(libbase::int32u key);

   // descriptor suite - low-level access
   void WriteScriptParameters();
   void ReadScriptParameters();

protected:
   // high-level access - reference creation/destruction
   PIActionReference MakeReference();
   void FreeReference(PIActionReference& reference);
   // high-level access - handle creation/destruction
   Handle MakeAlias(const char *sPathName);
   void FreeAlias(Handle& alias);
   // high-level access - descriptor creation/destruction
   PIActionDescriptor MakeDescriptor();
   void FreeDescriptor(PIActionDescriptor& descriptor);

   // high-level access - reference fill-in
   void PutEnumerated(PIActionReference reference, DescriptorClassID desiredClass, DescriptorEnumTypeID type, DescriptorEnumID value);
   void PutOffset(PIActionReference reference, DescriptorClassID desiredClass, int32 value);
   // high-level access - descriptor fill-in
   void PutInteger(PIActionDescriptor descriptor, DescriptorKeyID key, int data);
   void PutFloat(PIActionDescriptor descriptor, DescriptorKeyID key, double data);
   void PutBoolean(PIActionDescriptor descriptor, DescriptorKeyID key, bool data);
   void PutString(PIActionDescriptor descriptor, DescriptorKeyID key, const char *data);
   void PutAlias(PIActionDescriptor descriptor, DescriptorKeyID key, Handle alias);
   void PutReference(PIActionDescriptor descriptor, DescriptorKeyID key, PIActionReference reference);
   void PutEnumerated(PIActionDescriptor descriptor, DescriptorKeyID key, DescriptorEnumTypeID type, DescriptorEnumID value);
   void PutObject(PIActionDescriptor descriptor, DescriptorKeyID key, DescriptorClassID type,  PIActionDescriptor value);
   // high-level access - descriptor read-out
   bool GetInteger(PIActionDescriptor descriptor, DescriptorKeyID key, int *data);
   bool GetFloat(PIActionDescriptor descriptor, DescriptorKeyID key, double *data);
   bool GetBoolean(PIActionDescriptor descriptor, DescriptorKeyID key, bool *data);
   bool GetString(PIActionDescriptor descriptor, DescriptorKeyID key, char *data);

   // high-level access - event playback
   PIActionDescriptor PlayEvent(DescriptorEventID event, PIActionDescriptor descriptor, PIDialogPlayOptions options);

   // event playback - steganography
   void PlayeventFilterATM(int nRadius, int nAlpha, bool bKeepNoise);
   void PlayeventFilterAW(int nRadius, double dNoise, bool bKeepNoise);
   void PlayeventFilterWavelet(int nWaveletType, int nWaveletPar, int nWaveletLevel, int nThreshType, int nThreshSelector, double dThreshCutoff, int nTileX, int nTileY, bool bWholeImage, bool bKeepNoise);
   void PlayeventFilterVariance(int nRadius, int nScale);
   void PlayeventFilterEnergy(const char *sFileName, bool bAppend, bool bDisplayVariance, bool bDisplayEnergy, bool bDisplayPixelCount);
   void PlayeventFilterExport(const char *sPathName);
   void PlayeventFilterEmbed(int nEmbedSeed, int nEmbedRate, double dEmbedStrength, bool bInterleave, int nInterleaverSeed, double dInterleaverDensity, int nSourceType, int nSourceSeed, const char *sSource, const char *sCodec);
   void PlayeventFilterExtract(int nEmbedSeed, int nEmbedRate, double dEmbedStrength, bool bPresetStrength, bool bInterleave, int nInterleaverSeed, double dInterleaverDensity, int nSourceType, int nSourceSeed, const char *sSource, const char *sCodec, const char *sResults, const char *sEmbedded, const char *sExtracted, const char *sUniform, const char *sDecoded, bool bPrintBER, bool bPrintSNR, bool bPrintEstimate, bool bPrintChiSquare, int nFeedback);

   // event playback - binary
   void PlayeventFilterOrphans(int nWeight, bool bKeepNoise);

   // event playback - photoshop
   void PlayeventOpen(const char *sPathName);
   void PlayeventClose(bool bSave);
   void PlayeventConvertMode(int nDepth);
   void PlayeventSelectState(int nOffset);
   void PlayeventSaveJPEG(int nJpegQ, const char *sPathName, bool bCopy=false);
   void PlayeventSaveLZW(const char *sPathName, bool bCopy=false);
   void PlayeventRevert();

   // virtual overrides - scripting
   virtual void WriteScriptParameters(PIActionDescriptor descriptor) {};
   virtual void ReadScriptParameters(PIActionDescriptor descriptor) {};

   // virtual overrides - data handling
   virtual void InitParameters() = 0;

   // virtual overrides - other
   virtual void ShowDialog(void) = 0;

public:
   // functions for use by clients (such as user-interface dialog classes)

public:
   // creation / destruction
        CPSAutomate();
        virtual ~CPSAutomate();

   // plug-in main function
   SPErr Main(const char* sCaller, const char* sSelector, void* pData);

   // plug-in entry & exit
   void Entry(void* pData);
   void Exit();

   // virtual overrides - plug-in interface
   virtual void Startup(void);
   virtual void Shutdown(void);
   virtual void Reload(void);
   virtual void Unload(void);
   virtual void About(void);
   virtual void Execute(void);
   virtual void Process(void);
};

} // end namespace

#endif

#endif
