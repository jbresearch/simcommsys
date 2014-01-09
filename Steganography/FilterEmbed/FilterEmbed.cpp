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
#include "FilterEmbed.h"
#include "FilterEmbedDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "limiter.h"
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedApp

BEGIN_MESSAGE_MAP(CFilterEmbedApp, CWinApp)
//{{AFX_MSG_MAP(CFilterEmbedApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedApp construction

CFilterEmbedApp::CFilterEmbedApp() : CPSPlugIn(sizeof(SFilterEmbedData), 131)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedApp filter selector functions

// show the about dialog here
void CFilterEmbedApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterEmbedApp::FilterStart(void)
   {
   // set up library names
   using libbase::trace;
   using libbase::vector;

   // tile the image row by row, keeping the same tile area as suggested
   SetTileWidth(GetImageWidth());
   SetTileHeight(max(1,GetSuggestedTileHeight()*GetSuggestedTileWidth()/GetImageWidth()));
   // show dialog if necessary, set up first tile, start progress indicator & timer
   CPSPlugIn::FilterStart();

   // load codec & puncture files as necessary
   LoadErrorControl(m_sData->sCodec, m_sData->sPuncture);

   // load/create data sequence
   vector<int> d(GetDataSize(m_sData->dInterleaverDensity, m_sData->nEmbedRate));
   switch(m_sData->nSourceType)
      {
      case 0:
         d = 0;
         break;
      case 1:
         d = (1<<GetDataWidth())-1;
         break;
      case 2:
         GenerateSourceSequence(d, GetDataWidth(), m_sData->nSourceSeed);
         break;
      case 3:
         LoadDataFile(m_sData->sSource, d, GetDataWidth());
         break;
      default:
         throw("Unsupported source type.");
         break;
      }
   // encode & puncture it, if necessary
   if(CodecPresent())
      {
      const vector<int> e = d;
      d.init(GetRawSize(m_sData->dInterleaverDensity)/m_sData->nEmbedRate);
      EncodeData(e, d);
      }
   // do bandwidth expansion, if necessary
   if(m_sData->nEmbedRate > 1)
      {
      const vector<int> e = d;
      d.init(GetRawSize(m_sData->dInterleaverDensity));
      BandwidthExpander(m_sData->nEmbedRate, e, d);
      }
   // create pseudo-noise sequence
   m_vdMessage.init(d.size());
   GenerateEmbedSequence(m_vdMessage, m_sData->nEmbedSeed);
   // modulate the sequence
   ModulateEmbedSequence(d, m_vdMessage, m_vdMessage);
   trace << "Uniform message mean = " << m_vdMessage.mean() << ", min = " << m_vdMessage.min() << ", max = " << m_vdMessage.max() << "\n";
   // convert to gaussian
   ConvertToGaussian(m_vdMessage, m_vdMessage);
   trace << "Gaussian message mean = " << m_vdMessage.mean() << ", std = " << m_vdMessage.sigma() << "\n";
   // scale gaussian sequence
   m_vdMessage *= pow(10.0, m_sData->dEmbedStrength/20.0);
   trace << "Scaled Gaussian message mean = " << m_vdMessage.mean() << ", std = " << m_vdMessage.sigma() << "\n";

   // interleave if necessary
   if(m_sData->bInterleave)
      {
      vector<int> viIndex;
      GenerateInterleaver(viIndex, GetImagePixels(), m_vdMessage.size(), m_sData->nInterleaverSeed);
      const vector<double> vdCopy = m_vdMessage;
      InterleaveMessage(viIndex, vdCopy, m_vdMessage);
      }
   }

void CFilterEmbedApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0, 100, 3, 4);

   // convert tile to matrix
   libbase::matrix<double> m;
   GetPixelMatrix(m);

   // embed in image
   int k = GetCurrentPlane() * GetImageWidth() * GetImageHeight() + GetCurrentCoordTop() * GetImageWidth();
   for(int j=0; j<m.size().cols(); j++)
      for(int i=0; i<m.size().rows(); i++)
         m(i,j) += m_vdMessage(k++);

   // clip & convert matrix to tile
   libimage::limiter<double> lim(0,1);
   lim.process(m);
   SetPixelMatrix(m);

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   }

void CFilterEmbedApp::FilterFinish(void)
   {
   // clean up memory usage
   m_vdMessage.init(0);
   FreeErrorControl();
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedApp helper functions

void CFilterEmbedApp::ShowDialog(void)
   {
   CFilterEmbedDlg   dlg;

   dlg.m_pPSPlugIn = this;

   // embedding system
   dlg.m_nEmbedSeed = m_sData->nEmbedSeed;
   dlg.m_nEmbedRate = m_sData->nEmbedRate;
   dlg.m_dEmbedStrength = m_sData->dEmbedStrength;
   // channel interleaver
   dlg.m_bInterleave = m_sData->bInterleave;
   dlg.m_nInterleaverSeed = m_sData->nInterleaverSeed;
   dlg.m_dInterleaverDensity = m_sData->dInterleaverDensity;
   // source data
   dlg.m_nSourceType = m_sData->nSourceType;
   dlg.m_nSourceSeed = m_sData->nSourceSeed;
   dlg.m_sSource = m_sData->sSource;
   // codec and puncture pattern
   dlg.m_sCodec = m_sData->sCodec;
   dlg.m_sPuncture = m_sData->sPuncture;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   // embedding system
   m_sData->nEmbedSeed = dlg.m_nEmbedSeed;
   m_sData->nEmbedRate = dlg.m_nEmbedRate;
   m_sData->dEmbedStrength = dlg.m_dEmbedStrength;
   // channel interleaver
   m_sData->bInterleave = dlg.m_bInterleave != 0;
   m_sData->nInterleaverSeed = dlg.m_nInterleaverSeed;
   m_sData->dInterleaverDensity = dlg.m_dInterleaverDensity;
   // source data
   m_sData->nSourceType = dlg.m_nSourceType;
   m_sData->nSourceSeed = dlg.m_nSourceSeed;
   strcpy(m_sData->sSource, dlg.m_sSource);
   // codec and puncture pattern
   strcpy(m_sData->sCodec, dlg.m_sCodec);
   strcpy(m_sData->sPuncture, dlg.m_sPuncture);
   SetShowDialog(false);
   }

void CFilterEmbedApp::InitPointer(char* sData)
   {
   m_sData = (SFilterEmbedData *) sData;
   }

void CFilterEmbedApp::InitParameters()
   {
   // embedding system
   m_sData->nEmbedSeed = 0;
   m_sData->nEmbedRate = 1;
   m_sData->dEmbedStrength = -30;
   // channel interleaver
   m_sData->bInterleave = true;
   m_sData->nInterleaverSeed = 0;
   m_sData->dInterleaverDensity = 1.0;
   // source data
   m_sData->nSourceType = 0;
   m_sData->nSourceSeed = 1;
   strcpy(m_sData->sSource, "");
   // codec and puncture pattern
   strcpy(m_sData->sCodec, "");
   strcpy(m_sData->sPuncture, "");
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedApp scripting support

void CFilterEmbedApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      // embedding system
      case keyEmbedSeed:
         GetInteger(token, &m_sData->nEmbedSeed);
         break;
      case keyEmbedRate:
         GetInteger(token, &m_sData->nEmbedRate);
         break;
      case keyEmbedStrength:
         GetFloat(token, &m_sData->dEmbedStrength);
         break;
         // channel interleaver
      case keyInterleave:
         GetBoolean(token, &m_sData->bInterleave);
         break;
      case keyInterleaverSeed:
         GetInteger(token, &m_sData->nInterleaverSeed);
         break;
      case keyInterleaverDensity:
         GetFloat(token, &m_sData->dInterleaverDensity);
         break;
         // source data
      case keySourceType:
         GetInteger(token, &m_sData->nSourceType);
         break;
      case keySourceSeed:
         GetInteger(token, &m_sData->nSourceSeed);
         break;
      case keySource:
         GetString(token, m_sData->sSource);
         break;
         // codec and puncture pattern
      case keyCodec:
         GetString(token, m_sData->sCodec);
         break;
      case keyPuncture:
         GetString(token, m_sData->sPuncture);
         break;
      // unknown
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterEmbedApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   // embedding system
   PutInteger(token, keyEmbedSeed, m_sData->nEmbedSeed);
   PutInteger(token, keyEmbedRate, m_sData->nEmbedRate);
   PutFloat(token, keyEmbedStrength, m_sData->dEmbedStrength);
   // channel interleaver
   PutBoolean(token, keyInterleave, m_sData->bInterleave);
   if(m_sData->bInterleave)
      {
      PutInteger(token, keyInterleaverSeed, m_sData->nInterleaverSeed);
      PutFloat(token, keyInterleaverDensity, m_sData->dInterleaverDensity);
      }
   // source data
   PutInteger(token, keySourceType, m_sData->nSourceType);
   PutInteger(token, keySourceSeed, m_sData->nSourceSeed);
   if(strlen(m_sData->sSource) > 0)
      PutString(token, keySource, m_sData->sSource);
   // codec and puncture pattern
   if(strlen(m_sData->sCodec) > 0)
      PutString(token, keyCodec, m_sData->sCodec);
   if(strlen(m_sData->sPuncture) > 0)
      PutString(token, keyPuncture, m_sData->sPuncture);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterEmbedApp object

CFilterEmbedApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
