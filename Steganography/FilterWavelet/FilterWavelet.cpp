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
#include "FilterWavelet.h"
#include "FilterWaveletDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "itfunc.h"
#include "limiter.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletApp

BEGIN_MESSAGE_MAP(CFilterWaveletApp, CWinApp)
//{{AFX_MSG_MAP(CFilterWaveletApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletApp construction

CFilterWaveletApp::CFilterWaveletApp() : CPSPlugIn(sizeof(SFilterWaveletData), 113)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletApp filter selector functions

// show the about dialog here
void CFilterWaveletApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterWaveletApp::FilterStart(void)
   {
   // FilterStart will get user parameters if necessary & select the first tile
   CPSPlugIn::FilterStart();
   // set up modified tile sizes
   SetTileWidth(m_sData->bWholeImage ? GetImageWidth() : min(m_sData->nTileX, GetImageWidth()));
   SetTileHeight(m_sData->bWholeImage ? GetImageHeight() : min(m_sData->nTileY, GetImageHeight()));
   // select the first tile again
   CPSPlugIn::IterationStart();

   // initialize the wavelet filter & current iteration
   waveletfilter::init(m_sData->nWaveletType, m_sData->nWaveletPar, m_sData->nWaveletLevel, m_sData->nThreshType, m_sData->nThreshSelector, m_sData->dThreshCutoff);
   waveletfilter::reset();
   m_nIteration = 0;
   }

void CFilterWaveletApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0, 100, m_nIteration, 2);

   // set up library names
   using libbase::matrix;
   using libbase::weight;
   using libbase::log2;
   using libimage::limiter;

   // do the processing for this tile
   matrix<double> in, out;
   GetPixelMatrix(in);

   if(m_nIteration == 0)
      {
      // first iteration: update statistics
      if(weight(in.size().rows()) != 1 || weight(in.size().cols()) != 1)
         {
         matrix<double> temp(1<<int(ceil(log2(in.size().rows()))), 1<<int(ceil(log2(in.size().cols()))));
         temp = 0;
         temp.copyfrom(in);
         waveletfilter::update(temp);
         }
      else
         waveletfilter::update(in);
      }
   else
      {
      // second iteration: do the wavelet shrinkage
      if(weight(in.size().rows()) != 1 || weight(in.size().cols()) != 1)
         {
         matrix<double> temp(1<<int(ceil(log2(in.size().rows()))), 1<<int(ceil(log2(in.size().cols()))));
         temp = 0;
         temp.copyfrom(in);
         waveletfilter::process(temp, temp);
         out.init(in.size());
         out.copyfrom(temp);
         }
      else
         waveletfilter::process(in, out);

      limiter<double> lim(0,1);
      lim.process(out);
      if(m_sData->bKeepNoise)
         {
         out *= -1;
         out += in;
         out += 0.5;
         lim.process(out);
         }
      SetPixelMatrix(out);
      }

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   // if we need to do a second iteration
   if(m_nIteration==0 && IterationDone())
      {
      // update iteration counter & restart from first tile
      m_nIteration++;
      IterationStart();
      // compute threshold from gathered statistics
      waveletfilter::estimate();
      }
   }

void CFilterWaveletApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletApp helper functions

void CFilterWaveletApp::ShowDialog(void)
   {
   CFilterWaveletDlg   dlg;

   dlg.m_pPSPlugIn = this;

   // wavelet basis
   dlg.m_nWaveletType = m_sData->nWaveletType;
   dlg.m_nWaveletPar = m_sData->nWaveletPar;
   dlg.m_nWaveletLevel = m_sData->nWaveletLevel;
   // thresholding
   dlg.m_nThreshType = m_sData->nThreshType;
   dlg.m_nThreshSelector = m_sData->nThreshSelector;
   dlg.m_dThreshCutoff = m_sData->dThreshCutoff;
   // tiling
   dlg.m_nTileX = (m_sData->nTileX == 0) ? GetSuggestedTileWidth() : m_sData->nTileX;
   dlg.m_nTileY = (m_sData->nTileY == 0) ? GetSuggestedTileHeight() : m_sData->nTileY;
   dlg.m_bWholeImage = m_sData->bWholeImage;
   // other
   dlg.m_bKeepNoise = m_sData->bKeepNoise;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   // wavelet basis
   m_sData->nWaveletType = dlg.m_nWaveletType;
   m_sData->nWaveletPar = dlg.m_nWaveletPar;
   m_sData->nWaveletLevel = dlg.m_nWaveletLevel;
   // thresholding
   m_sData->nThreshType = dlg.m_nThreshType;
   m_sData->nThreshSelector = dlg.m_nThreshSelector;
   m_sData->dThreshCutoff = dlg.m_dThreshCutoff;
   // tiling
   m_sData->nTileX = dlg.m_nTileX;
   m_sData->nTileY = dlg.m_nTileY;
   m_sData->bWholeImage = dlg.m_bWholeImage != 0;
   // other
   m_sData->bKeepNoise = dlg.m_bKeepNoise != 0;

   SetShowDialog(false);
   }

void CFilterWaveletApp::InitPointer(char* sData)
   {
   m_sData = (SFilterWaveletData *) sData;
   }

void CFilterWaveletApp::InitParameters()
   {
   // wavelet basis
   m_sData->nWaveletType = 0;
   m_sData->nWaveletPar = 0;
   m_sData->nWaveletLevel = 5;
   // thresholding
   m_sData->nThreshType = 0;
   m_sData->nThreshSelector = 0;
   m_sData->dThreshCutoff = 0.95;
   // tiling
   m_sData->nTileX = 0; // special value - replaced with Photoshop suggestions on first call
   m_sData->nTileY = 0; // idem
   m_sData->bWholeImage = true;
   // other
   m_sData->bKeepNoise = false;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletApp scripting support

void CFilterWaveletApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      // wavelet basis
      case keyWaveletType:
         GetInteger(token, &m_sData->nWaveletType);
         break;
      case keyWaveletPar:
         GetInteger(token, &m_sData->nWaveletPar);
         break;
      case keyWaveletLevel:
         GetInteger(token, &m_sData->nWaveletLevel);
         break;
      // thresholding
      case keyThreshType:
         GetInteger(token, &m_sData->nThreshType);
         break;
      case keyThreshSelector:
         GetInteger(token, &m_sData->nThreshSelector);
         break;
      case keyThreshCutoff:
         GetFloat(token, &m_sData->dThreshCutoff);
         break;
      // tiling
      case keyTileX:
         GetInteger(token, &m_sData->nTileX);
         break;
      case keyTileY:
         GetInteger(token, &m_sData->nTileY);
         break;
      case keyWholeImage:
         GetBoolean(token, &m_sData->bWholeImage);
         break;
      // other
      case keyKeepNoise:
         GetBoolean(token, &m_sData->bKeepNoise);
         break;
      // unknown
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterWaveletApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   // wavelet basis
   PutInteger(token, keyWaveletType, m_sData->nWaveletType);
   PutInteger(token, keyWaveletPar, m_sData->nWaveletPar);
   PutInteger(token, keyWaveletLevel, m_sData->nWaveletLevel);
   // thresholding
   PutInteger(token, keyThreshType, m_sData->nThreshType);
   PutInteger(token, keyThreshSelector, m_sData->nThreshSelector);
   PutFloat(token, keyThreshCutoff, m_sData->dThreshCutoff);
   // tiling
   if(!m_sData->bWholeImage)
      {
      PutInteger(token, keyTileX, m_sData->nTileX);
      PutInteger(token, keyTileY, m_sData->nTileY);
      PutBoolean(token, keyWholeImage, m_sData->bWholeImage);
      }
   // other
   if(m_sData->bKeepNoise)
      PutBoolean(token, keyKeepNoise, m_sData->bKeepNoise);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterWaveletApp object

CFilterWaveletApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
