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

#include "stdafx.h"
#include "PSAutomate.h"
#include "ScriptingKeys.h"

#ifdef ADOBESDK

#include "PIGeneral.h"
#include "PIActions.h"
#include "PIActionsPlugIn.h"
#include "SPInterf.h"
#include "SPBasic.h"
#include "SPAccess.h"

namespace libwin {

using libbase::trace;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CPSAutomate::CPSAutomate()
   {
   }

CPSAutomate::~CPSAutomate()
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate main function

SPErr CPSAutomate::Main(const char* sCaller, const char* sSelector, void* pData)
   {
   AFX_MANAGE_STATE(AfxGetStaticModuleState());
   try
      {
      // acquire suites
      Entry(pData);

      // interface callers
      if(m_pBasic->IsEqual((char*)sCaller, kSPInterfaceCaller))
         {
         // startup
         if(m_pBasic->IsEqual((char*)sSelector, kSPInterfaceStartupSelector))
            Startup();
         // shutdown
         else if(m_pBasic->IsEqual((char*)sSelector, kSPInterfaceShutdownSelector))
            Shutdown();
         // about
         else if(m_pBasic->IsEqual((char*)sSelector, kSPInterfaceAboutSelector))
            About();
         else
            trace << "Error: Unknown Interface selector \'" << sSelector << "\'.\n";
         }
      // access callers
      else if(m_pBasic->IsEqual((char*)sCaller, kSPAccessCaller))
         {
         // reload
         if(m_pBasic->IsEqual((char*)sSelector, kSPAccessReloadSelector))
            Reload();
         // unload
         else if(m_pBasic->IsEqual((char*)sSelector, kSPAccessUnloadSelector))
            Unload();
         else
            trace << "Error: Unknown Access selector \'" << sSelector << "\'.\n";
         }
      // photoshop callers
      else if(m_pBasic->IsEqual((char*)sCaller, kPSPhotoshopCaller))
         {
         // core functionality
         if(m_pBasic->IsEqual((char*)sSelector, kPSDoIt))
            Execute();
         else
            trace << "Error: Unknown Photoshop selector \'" << sSelector << "\'.\n";
         }
      else
         trace << "Error: Unknown caller \'" << sCaller << "\'.\n";

      // release suites and return
      Exit();
      return kSPNoError;
      }

   catch(SPErr error)
      {
      return error;
      }

   catch(...)
      {
      return kSPUnimplementedError;
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate entry and exit points

// this gets called every time through the main() loop so we stay in sync
void CPSAutomate::Entry(void* pData)
   {
   TRACE("Automate Entry\n");

   // store entry message block
   m_pData = pData;

   // get pointer to basic suite
   SPMessageData *pMessageData = (SPMessageData *)m_pData;
   m_pBasic = pMessageData->basic;

   // acquire suites
   if(m_pBasic->AcquireSuite(kPSActionDescriptorSuite, kPSActionDescriptorSuiteVersion, (void **)&m_pActionDescriptor))
      throw(errPlugInHostInsufficient);
   if(m_pBasic->AcquireSuite(kPSActionReferenceSuite, kPSActionReferenceSuiteVersion, (void **)&m_pActionReference))
      throw(errPlugInHostInsufficient);
   if(m_pBasic->AcquireSuite(kPSActionControlSuite, kPSActionControlSuitePrevVersion, (void **)&m_pActionControl))
      throw(errPlugInHostInsufficient);
   if(m_pBasic->AcquireSuite(kPSActionListSuite, kPSActionListSuiteVersion, (void **)&m_pActionList))
      throw(errPlugInHostInsufficient);
   if(m_pBasic->AcquireSuite(kPSHandleSuite, kPSHandleSuiteVersion1, (void **)&m_pHandle))
      throw(errPlugInHostInsufficient);
   }

// this gets called every time through the main() loop so we stay in sync
void CPSAutomate::Exit()
   {
   TRACE("Automate Exit\n");

   // release suites
   m_pBasic->ReleaseSuite(kPSActionDescriptorSuite, kPSActionDescriptorSuiteVersion);
   m_pBasic->ReleaseSuite(kPSActionReferenceSuite, kPSActionReferenceSuiteVersion);
   m_pBasic->ReleaseSuite(kPSActionControlSuite, kPSActionControlSuitePrevVersion);
   m_pBasic->ReleaseSuite(kPSActionListSuite, kPSActionListSuiteVersion);
   m_pBasic->ReleaseSuite(kPSHandleSuite, kPSHandleSuiteVersion1);
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate plug-in interface functions

// allocate memory and do other initialization tasks here
void CPSAutomate::Startup(void)
   {
   TRACE("Automate Startup\n");
   }

// release memory and do other finalization tasks here
void CPSAutomate::Shutdown(void)
   {
   TRACE("Automate Shutdown\n");
   }

// restore state information
void CPSAutomate::Reload(void)
   {
   TRACE("Automate Reload\n");
   }

// save state information
void CPSAutomate::Unload(void)
   {
   TRACE("Automate Unload\n");
   }

// show the about dialog here
void CPSAutomate::About(void)
   {
   TRACE("Automate About\n");
   }

void CPSAutomate::Execute(void)
   {
   TRACE("Automate Execute\n");
   // initialize parameters
   InitParameters();
   // read parameters from the scripting sub-system, if given
   ReadScriptParameters();
   // show user dialog to get parameters
   ShowDialog();
   // start operation timer
   m_tDuration.start();
   // call derived class function
   Process();
   // stop operation timer
   m_tDuration.stop();
   TRACE("Time taken: %s\n", std::string(m_tDuration).c_str());
   // write scripting parameters
   WriteScriptParameters();
   }

void CPSAutomate::Process(void)
   {
   TRACE("Automate Process\n");
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate private helpers - internal utility functions

CString CPSAutomate::KeyToString(libbase::int32u key)
   {
   CString sTemp;
   sTemp.Format("\'%c%c%c%c\'", key>>24, key>>16, key>>8, key>>0);
   return sTemp;
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate private helpers - descriptor suite low-level access

void CPSAutomate::ReadScriptParameters()
   {
   PSActionsPlugInMessage* pActionsPlugInMessage = (PSActionsPlugInMessage *)m_pData;
   PIActionParameters *pActionParameters = pActionsPlugInMessage->actionParameters;
   PIActionDescriptor descriptor = pActionParameters->descriptor;
   if(descriptor != NULL)
      {
      TRACE("Automate Reading Script Parameters\n");
      // make sure we're starting with the default values
      InitParameters();
      // get all the keys
      ReadScriptParameters(descriptor);
      }
   }

void CPSAutomate::WriteScriptParameters()
   {
   PSActionsPlugInMessage* pActionsPlugInMessage = (PSActionsPlugInMessage *)m_pData;
   PIActionParameters *pActionParameters = pActionsPlugInMessage->actionParameters;
   PIActionDescriptor descriptor = MakeDescriptor();
   if(descriptor != NULL)
      {
      TRACE("Filter Writing Script Parameters\n");
      WriteScriptParameters(descriptor);
      pActionParameters->descriptor = descriptor;
      pActionParameters->recordInfo = plugInDialogOptional;
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate high-level access

// reference creation/destruction

PIActionReference CPSAutomate::MakeReference()
   {
   TRACE("Automate MakeReference()\n");
   PIActionReference reference = NULL;
   if(SPErr error = m_pActionReference->Make(&reference))
      {
      FreeReference(reference);
      throw(error);
      }
   return reference;
   }

void CPSAutomate::FreeReference(PIActionReference& reference)
   {
   TRACE("Automate FreeReference(0x%x)\n", (void *)reference);
   if(reference != NULL)
      {
      m_pActionReference->Free(reference);
      reference = NULL;
      }
   }

// handle creation/destruction

Handle CPSAutomate::MakeAlias(const char *sPathName)
   {
   TRACE("Automate MakeAlias(\"%s\")\n", sPathName);
   Handle alias = m_pHandle->New(strlen(sPathName)+1);
   if(alias == NULL)
      throw((SPErr)kSPLogicError);
   Boolean oldLock;
   Ptr address;
   m_pHandle->SetLock(alias, true, &address, &oldLock);
   strncpy(address, sPathName, strlen(sPathName)+1);
   m_pHandle->SetLock(alias, false, &address, &oldLock);
   return alias;
   }

void CPSAutomate::FreeAlias(Handle& alias)
   {
   TRACE("Automate FreeAlias(0x%x)\n", (void *)alias);
   if(alias != NULL)
      {
      m_pHandle->Dispose(alias);
      alias = NULL;
      }
   }

// descriptor creation/destruction

PIActionDescriptor CPSAutomate::MakeDescriptor()
   {
   TRACE("Automate MakeDescriptor()\n");
   PIActionDescriptor descriptor = NULL;
   if(SPErr error = m_pActionDescriptor->Make(&descriptor))
      {
      FreeDescriptor(descriptor);
      throw(error);
      }
   return descriptor;
   }

void CPSAutomate::FreeDescriptor(PIActionDescriptor& descriptor)
   {
   TRACE("Automate FreeDescriptor(0x%x)\n", (void *)descriptor);
   if(descriptor != NULL)
      {
      m_pActionDescriptor->Free(descriptor);
      descriptor = NULL;
      }
   }

// high-level access - reference fill-in

void CPSAutomate::PutEnumerated(PIActionReference reference, DescriptorClassID desiredClass, DescriptorEnumTypeID type, DescriptorEnumID value)
   {
   TRACE("Automate PutEnumerated[ref](0x%x, %s, %s, %s)\n", (void *)reference, KeyToString(desiredClass), KeyToString(type), KeyToString(value));
   if(SPErr error = m_pActionReference->PutEnumerated(reference, desiredClass, type, value))
      {
      FreeReference(reference);
      throw(error);
      }
   }

void CPSAutomate::PutOffset(PIActionReference reference, DescriptorClassID desiredClass, int32 value)
   {
   TRACE("Automate PutOffset[ref](0x%x, %s, %d)\n", (void *)reference, KeyToString(desiredClass), value);
   if(SPErr error = m_pActionReference->PutOffset(reference, desiredClass, value))
      {
      FreeReference(reference);
      throw(error);
      }
   }

// high-level access - descriptor fill-in

void CPSAutomate::PutInteger(PIActionDescriptor descriptor, DescriptorKeyID key, int data)
   {
   TRACE("Automate PutInteger(0x%x, %s, %d)\n", (void *)descriptor, KeyToString(key), data);
   if(SPErr error = m_pActionDescriptor->PutInteger(descriptor, key, data))
      {
      FreeDescriptor(descriptor);
      throw(error);
      }
   }

void CPSAutomate::PutFloat(PIActionDescriptor descriptor, DescriptorKeyID key, double data)
   {
   TRACE("Automate PutFloat(0x%x, %s, %g)\n", (void *)descriptor, KeyToString(key), data);
   if(SPErr error = m_pActionDescriptor->PutFloat(descriptor, key, data))
      {
      FreeDescriptor(descriptor);
      throw(error);
      }
   }

void CPSAutomate::PutBoolean(PIActionDescriptor descriptor, DescriptorKeyID key, bool data)
   {
   TRACE("Automate PutBoolean(0x%x, %s, %s)\n", (void *)descriptor, KeyToString(key), data ? "true" : "false");
   if(SPErr error = m_pActionDescriptor->PutBoolean(descriptor, key, data))
      {
      FreeDescriptor(descriptor);
      throw(error);
      }
   }

void CPSAutomate::PutString(PIActionDescriptor descriptor, DescriptorKeyID key, const char *data)
   {
   TRACE("Automate PutString(0x%x, %s, %s)\n", (void *)descriptor, KeyToString(key), data);
   if(SPErr error = m_pActionDescriptor->PutString(descriptor, key, (char *)data))
      {
      FreeDescriptor(descriptor);
      throw(error);
      }
   }

void CPSAutomate::PutAlias(PIActionDescriptor descriptor, DescriptorKeyID key, Handle alias)
   {
   TRACE("Automate PutAlias(0x%x, %s, 0x%x)\n", (void *)descriptor, KeyToString(key), (void *)alias);
   if(SPErr error = m_pActionDescriptor->PutAlias(descriptor, key, alias))
      {
      FreeDescriptor(descriptor);
      FreeAlias(alias);
      throw(error);
      }
   }

void CPSAutomate::PutReference(PIActionDescriptor descriptor, DescriptorKeyID key, PIActionReference reference)
   {
   TRACE("Automate PutReference(0x%x, %s, 0x%x)\n", (void *)descriptor, KeyToString(key), (void *)reference);
   if(SPErr error = m_pActionDescriptor->PutReference(descriptor, key, reference))
      {
      FreeDescriptor(descriptor);
      FreeReference(reference);
      throw(error);
      }
   }

void CPSAutomate::PutEnumerated(PIActionDescriptor descriptor, DescriptorKeyID key, DescriptorEnumTypeID type, DescriptorEnumID value)
   {
   TRACE("Automate PutEnumerated(0x%x, %s, %s, %s)\n", (void *)descriptor, KeyToString(key), KeyToString(type), KeyToString(value));
   if(SPErr error = m_pActionDescriptor->PutEnumerated(descriptor, key, type, value))
      {
      FreeDescriptor(descriptor);
      throw(error);
      }
   }

void CPSAutomate::PutObject(PIActionDescriptor descriptor, DescriptorKeyID key, DescriptorClassID type,  PIActionDescriptor value)
   {
   TRACE("Automate PutObject(0x%x, %s, %s, 0x%x)\n", (void *)descriptor, KeyToString(key), KeyToString(type), (void *)value);
   if(SPErr error = m_pActionDescriptor->PutObject(descriptor, key, type, value))
      {
      FreeDescriptor(descriptor);
      FreeDescriptor(value);
      throw(error);
      }
   }

// high-level access - descriptor read-out

bool CPSAutomate::GetInteger(PIActionDescriptor descriptor, DescriptorKeyID key, int *data)
   {
   int32 temp;
   if(SPErr error = m_pActionDescriptor->GetInteger(descriptor, key, &temp))
      return false;
   *data = temp;
   TRACE("Automate GetInteger(0x%x, %s) = %d\n", (void *)descriptor, KeyToString(key), *data);
   return true;
   }

bool CPSAutomate::GetFloat(PIActionDescriptor descriptor, DescriptorKeyID key, double *data)
   {
   double temp;
   if(SPErr error = m_pActionDescriptor->GetFloat(descriptor, key, &temp))
      return false;
   *data = temp;
   TRACE("Automate GetFloat(0x%x, %s) = %g\n", (void *)descriptor, KeyToString(key), *data);
   return true;
   }

bool CPSAutomate::GetBoolean(PIActionDescriptor descriptor, DescriptorKeyID key, bool *data)
   {
   Boolean temp;
   if(SPErr error = m_pActionDescriptor->GetBoolean(descriptor, key, &temp))
      return false;
   *data = temp!=0;
   TRACE("Automate GetBoolean(0x%x, %s) = %s\n", (void *)descriptor, KeyToString(key), *data ? "true" : "false");
   return true;
   }

bool CPSAutomate::GetString(PIActionDescriptor descriptor, DescriptorKeyID key, char *data)
   {
   char temp[256];
   if(SPErr error = m_pActionDescriptor->GetString(descriptor, key, temp, 256))
      return false;
   strcpy(data, temp);
   TRACE("Automate GetString(0x%x, %s) = %s\n", (void *)descriptor, KeyToString(key), data);
   return true;
   }

// high-level access - event playback

PIActionDescriptor CPSAutomate::PlayEvent(DescriptorEventID event, PIActionDescriptor descriptor, PIDialogPlayOptions options)
   {
   TRACE("Automate PlayEvent(%s, 0x%x, 0x%x)\n", KeyToString(event), (void *)descriptor, options);
   PIActionDescriptor result = NULL;
   if(SPErr error = m_pActionControl->Play(&result, event, descriptor, options))
      {
      FreeDescriptor(result);
      FreeDescriptor(descriptor);
      throw(error);
      }
   FreeDescriptor(descriptor);
   return result;
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate event playback - steganography

void CPSAutomate::PlayeventFilterATM(int nRadius, int nAlpha, bool bKeepNoise)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutInteger(descriptor, keyRadius, nRadius);
   PutInteger(descriptor, keyAlpha, nAlpha);
   PutBoolean(descriptor, keyKeepNoise, bKeepNoise);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterATM, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterAW(int nRadius, double dNoise, bool bKeepNoise)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutInteger(descriptor, keyRadius, nRadius);
   PutFloat(descriptor, keyNoise, dNoise);
   PutBoolean(descriptor, keyKeepNoise, bKeepNoise);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterAW, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterWavelet(int nWaveletType, int nWaveletPar, int nWaveletLevel, int nThreshType, int nThreshSelector, double dThreshCutoff, int nTileX, int nTileY, bool bWholeImage, bool bKeepNoise)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // wavelet basis
   PutInteger(descriptor, keyWaveletType, nWaveletType);
   PutInteger(descriptor, keyWaveletPar, nWaveletPar);
   PutInteger(descriptor, keyWaveletLevel, nWaveletLevel);
   // thresholding
   PutInteger(descriptor, keyThreshType, nThreshType);
   PutInteger(descriptor, keyThreshSelector, nThreshSelector);
   PutFloat(descriptor, keyThreshCutoff, dThreshCutoff);
   // tiling
   PutInteger(descriptor, keyTileX, nTileX);
   PutInteger(descriptor, keyTileY, nTileY);
   PutBoolean(descriptor, keyWholeImage, bWholeImage);
   // other
   PutBoolean(descriptor, keyKeepNoise, bKeepNoise);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterWavelet, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterVariance(int nRadius, int nScale)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutInteger(descriptor, keyRadius, nRadius);
   PutInteger(descriptor, keyScale, nScale);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterVariance, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterEnergy(const char *sFileName, bool bAppend, bool bDisplayVariance, bool bDisplayEnergy, bool bDisplayPixelCount)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutString(descriptor, keyFileName, sFileName);
   PutBoolean(descriptor, keyAppend, bAppend);
   PutBoolean(descriptor, keyDisplayVariance, bDisplayVariance);
   PutBoolean(descriptor, keyDisplayEnergy, bDisplayEnergy);
   PutBoolean(descriptor, keyDisplayPixelCount, bDisplayPixelCount);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterEnergy, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterExport(const char *sPathName)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutString(descriptor, keyPathName, sPathName);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterExport, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterEmbed(int nEmbedSeed, int nEmbedRate, double dEmbedStrength, bool bInterleave, int nInterleaverSeed, double dInterleaverDensity, int nSourceType, int nSourceSeed, const char *sSource, const char *sCodec)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // embedding system
   PutInteger(descriptor, keyEmbedSeed, nEmbedSeed);
   PutInteger(descriptor, keyEmbedRate, nEmbedRate);
   PutFloat(descriptor, keyEmbedStrength, dEmbedStrength);
   // channel interleaver
   PutBoolean(descriptor, keyInterleave, bInterleave);
   PutInteger(descriptor, keyInterleaverSeed, nInterleaverSeed);
   PutFloat(descriptor, keyInterleaverDensity, dInterleaverDensity);
   // source data
   PutInteger(descriptor, keySourceType, nSourceType);
   PutInteger(descriptor, keySourceSeed, nSourceSeed);
   PutString(descriptor, keySource, sSource);
   // codec
   PutString(descriptor, keyCodec, sCodec);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterEmbed, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventFilterExtract(int nEmbedSeed, int nEmbedRate, double dEmbedStrength, bool bPresetStrength, bool bInterleave, int nInterleaverSeed, double dInterleaverDensity, int nSourceType, int nSourceSeed, const char *sSource, const char *sCodec, const char *sResults, const char *sEmbedded, const char *sExtracted, const char *sUniform, const char *sDecoded, bool bPrintBER, bool bPrintSNR, bool bPrintEstimate, bool bPrintChiSquare, int nFeedback)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // embedding system
   PutInteger(descriptor, keyEmbedSeed, nEmbedSeed);
   PutInteger(descriptor, keyEmbedRate, nEmbedRate);
   PutFloat(descriptor, keyEmbedStrength, dEmbedStrength);
   PutBoolean(descriptor, keyPresetStrength, bPresetStrength);
   // channel interleaver
   PutBoolean(descriptor, keyInterleave, bInterleave);
   PutInteger(descriptor, keyInterleaverSeed, nInterleaverSeed);
   PutFloat(descriptor, keyInterleaverDensity, dInterleaverDensity);
   // source data
   PutInteger(descriptor, keySourceType, nSourceType);
   PutInteger(descriptor, keySourceSeed, nSourceSeed);
   PutString(descriptor, keySource, sSource);
   // codec
   PutString(descriptor, keyCodec, sCodec);
   // results storage
   PutString(descriptor, keyResults, sResults);
   PutString(descriptor, keyEmbedded, sEmbedded);
   PutString(descriptor, keyExtracted, sExtracted);
   PutString(descriptor, keyUniform, sUniform);
   PutString(descriptor, keyDecoded, sDecoded);
   // channel parameter computation / feedback
   PutBoolean(descriptor, keyPrintBER, bPrintBER);
   PutBoolean(descriptor, keyPrintSNR, bPrintSNR);
   PutBoolean(descriptor, keyPrintEstimate, bPrintEstimate);
   PutBoolean(descriptor, keyPrintChiSquare, bPrintChiSquare);
   PutInteger(descriptor, keyFeedback, nFeedback);
   // play event
   PIActionDescriptor result = PlayEvent(eventFilterExtract, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

/////////////////////////////////////////////////////////////////////////////
// CPSAutomate event playback - photoshop

void CPSAutomate::PlayeventOpen(const char *sPathName)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   Handle alias = MakeAlias(sPathName);
   PutAlias(descriptor, keyNull, alias);
   // play event
   try
      {
      PIActionDescriptor result = PlayEvent(eventOpen, descriptor, plugInDialogSilent);
      FreeDescriptor(result);
      FreeAlias(alias);
      }
   catch(SPErr error)
      {
      FreeAlias(alias);
      throw(error);
      }
   }

void CPSAutomate::PlayeventClose(bool bSave)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutEnumerated(descriptor, keySaving, typeYesNo, bSave ? enumNo : enumYes);
   // play event
   PIActionDescriptor result = PlayEvent(eventClose, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventConvertMode(int nDepth)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PutInteger(descriptor, keyDepth, 16);
   // play event
   PIActionDescriptor result = PlayEvent(eventConvertMode, descriptor, plugInDialogSilent);
   FreeDescriptor(result);
   }

void CPSAutomate::PlayeventSelectState(int nOffset)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   // insert parameters
   PIActionReference reference = MakeReference();
   PutOffset(reference, classHistoryState, nOffset);
   PutReference(descriptor, keyNull, reference);
   // play event
   try
      {
      PIActionDescriptor result = PlayEvent(eventSelect, descriptor, plugInDialogSilent);
      FreeDescriptor(result);
      FreeReference(reference);
      }
   catch(SPErr error)
      {
      FreeReference(reference);
      throw(error);
      }
   }

void CPSAutomate::PlayeventSaveJPEG(int nJpegQ, const char *sPathName, bool bCopy)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   PIActionDescriptor settings = MakeDescriptor();
   Handle alias = MakeAlias(sPathName);
   try
      {
      // insert parameters
      PutEnumerated(settings, keyByteOrder, typePlatform, enumIBMPC);
      PutEnumerated(settings, keyEncoding, typeEncoding, enumJPEG);
      PutInteger(settings, keyExtendedQuality, nJpegQ);
      PutObject(descriptor, keyAs, classTIFFFormat, settings);
      PutAlias(descriptor, keyIn, alias);
      PutBoolean(descriptor, keyLowerCase, true);
      if(bCopy)
         {
         PutBoolean(descriptor, keyCopy, true);
         PutBoolean(descriptor, keyLayers, false);
         }
      // play event
      PIActionDescriptor result = PlayEvent(eventSave, descriptor, plugInDialogSilent);
      FreeDescriptor(result);
      FreeDescriptor(settings);
      FreeAlias(alias);
      }
   catch(SPErr error)
      {
      FreeDescriptor(descriptor);
      FreeDescriptor(settings);
      FreeAlias(alias);
      throw(error);
      }
   }

void CPSAutomate::PlayeventSaveLZW(const char *sPathName, bool bCopy)
   {
   PIActionDescriptor descriptor = MakeDescriptor();
   PIActionDescriptor settings = MakeDescriptor();
   Handle alias = MakeAlias(sPathName);
   try
      {
      // insert parameters
      PutEnumerated(settings, keyByteOrder, typePlatform, enumIBMPC);
      PutBoolean(settings, keyLZWCompression, true);
      PutObject(descriptor, keyAs, classTIFFFormat, settings);
      PutAlias(descriptor, keyIn, alias);
      PutBoolean(descriptor, keyLowerCase, true);
      if(bCopy)
         {
         PutBoolean(descriptor, keyCopy, true);
         PutBoolean(descriptor, keyLayers, false);
         }
      // play event
      PIActionDescriptor result = PlayEvent(eventSave, descriptor, plugInDialogSilent);
      FreeDescriptor(result);
      FreeDescriptor(settings);
      FreeAlias(alias);
      }
   catch(SPErr error)
      {
      FreeDescriptor(descriptor);
      FreeDescriptor(settings);
      FreeAlias(alias);
      throw(error);
      }
   }

void CPSAutomate::PlayeventRevert()
   {
   // play event
   PIActionDescriptor result = PlayEvent(eventRevert, NULL, plugInDialogSilent);
   FreeDescriptor(result);
   }

} // end namespace

#endif
