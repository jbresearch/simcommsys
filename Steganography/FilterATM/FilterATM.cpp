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
#include "FilterATM.h"
#include "FilterATMDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "limiter.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterATMApp

BEGIN_MESSAGE_MAP(CFilterATMApp, CWinApp)
//{{AFX_MSG_MAP(CFilterATMApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterATMApp construction

CFilterATMApp::CFilterATMApp() : CPSPlugIn(sizeof(SFilterATMData), 101)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterATMApp filter selector functions

// show the about dialog here
void CFilterATMApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterATMApp::FilterStart(void)
   {
   // FilterStart will get user parameters if necessary & select the first tile
   CPSPlugIn::FilterStart();
   // Once we have radius, setup overlap for future tile updates
   SetTileOverlap(2 * m_sData->nRadius);
   // initialize the alpha-trimmed mean filter
   atmfilter<double>::init(m_sData->nRadius, m_sData->nAlpha);
   }

void CFilterATMApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0);

   // do the processing for this tile
   libbase::matrix<double> in, out;
   GetPixelMatrix(in);
   atmfilter<double>::process(in, out);
   if(m_sData->bKeepNoise)
      {
      out *= -1;
      out += in;
      out += 0.5;
      libimage::limiter<double> lim(0,1);
      lim.process(out);
      }
   SetPixelMatrix(out);

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   }

void CFilterATMApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterATMApp helper functions

void CFilterATMApp::ShowDialog(void)
   {
   CFilterATMDlg   dlg;

   dlg.m_pPSPlugIn = this;

   dlg.m_nAlpha = m_sData->nAlpha;
   dlg.m_nRadius = m_sData->nRadius;
   dlg.m_bKeepNoise = m_sData->bKeepNoise;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   m_sData->nAlpha = dlg.m_nAlpha;
   m_sData->nRadius = dlg.m_nRadius;
   m_sData->bKeepNoise = dlg.m_bKeepNoise != 0;
   SetShowDialog(false);
   }

void CFilterATMApp::InitPointer(char* sData)
   {
   m_sData = (SFilterATMData *) sData;
   }

void CFilterATMApp::InitParameters()
   {
   m_sData->nAlpha = 1;
   m_sData->nRadius = 1;
   m_sData->bKeepNoise = false;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterATMApp scripting support

void CFilterATMApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      case keyRadius:
         GetInteger(token, &m_sData->nRadius);
         break;
      case keyAlpha:
         GetInteger(token, &m_sData->nAlpha);
         break;
      case keyKeepNoise:
         GetBoolean(token, &m_sData->bKeepNoise);
         break;
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterATMApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   PutInteger(token, keyRadius, m_sData->nRadius);
   PutInteger(token, keyAlpha, m_sData->nAlpha);
   if(m_sData->bKeepNoise)
      PutBoolean(token, keyKeepNoise, m_sData->bKeepNoise);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterATMApp object

CFilterATMApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
