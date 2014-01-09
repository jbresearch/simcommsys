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
#include "FilterAW.h"
#include "FilterAWDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "limiter.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterAWApp

BEGIN_MESSAGE_MAP(CFilterAWApp, CWinApp)
//{{AFX_MSG_MAP(CFilterAWApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterAWApp construction

CFilterAWApp::CFilterAWApp() : CPSPlugIn(sizeof(SFilterAWData), 110)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterAWApp filter selector functions

// show the about dialog here
void CFilterAWApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterAWApp::FilterStart(void)
   {
   // FilterStart will get user parameters if necessary & select the first tile
   CPSPlugIn::FilterStart();
   // Once we have radius, setup overlap for future tile updates
   SetTileOverlap(2 * m_sData->nRadius);
   // initialize the adaptive wiener filter & current iteration
   awfilter<double>::init(m_sData->nRadius, m_sData->dNoise);
   awfilter<double>::reset();
   m_nIteration = 0;
   }

void CFilterAWApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0, 100, m_nIteration, m_sData->bAuto ? 2 : 1);

   // get this tile for processing
   libbase::matrix<double> in, out;
   GetPixelMatrix(in);

   if(m_nIteration==0 && m_sData->bAuto)
      {
      // first iteration: update statistics
      awfilter<double>::update(in);
      }
   else
      {
      // second iteration: filter this tile & write back
      awfilter<double>::process(in, out);
      libimage::limiter<double> lim(0,1);
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
   if(m_nIteration==0 && m_sData->bAuto && IterationDone())
      {
      // update iteration counter & restart from first tile
      m_nIteration++;
      IterationStart();
      // compute threshold from gathered statistics
      awfilter<double>::estimate();
      m_sData->dNoise = awfilter<double>::get_estimate();
      }
   }

void CFilterAWApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterAWApp helper functions

void CFilterAWApp::ShowDialog(void)
   {
   CFilterAWDlg   dlg;

   dlg.m_pPSPlugIn = this;

   dlg.m_nRadius = m_sData->nRadius;
   dlg.m_dNoise = m_sData->dNoise;
   dlg.m_bAuto = m_sData->bAuto;
   dlg.m_bKeepNoise = m_sData->bKeepNoise;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   m_sData->nRadius = dlg.m_nRadius;
   m_sData->dNoise = dlg.m_dNoise;
   m_sData->bAuto = dlg.m_bAuto != 0;
   m_sData->bKeepNoise = dlg.m_bKeepNoise != 0;
   SetShowDialog(false);
   }

void CFilterAWApp::InitPointer(char* sData)
   {
   m_sData = (SFilterAWData *) sData;
   }

void CFilterAWApp::InitParameters()
   {
   m_sData->nRadius = 1;
   m_sData->dNoise = 0;
   m_sData->bAuto = true;
   m_sData->bKeepNoise = false;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterAWApp scripting support

void CFilterAWApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      case keyRadius:
         GetInteger(token, &m_sData->nRadius);
         break;
      case keyNoise:
         GetFloat(token, &m_sData->dNoise);
         break;
      case keyAuto:
         GetBoolean(token, &m_sData->bAuto);
         break;
      case keyKeepNoise:
         GetBoolean(token, &m_sData->bKeepNoise);
         break;
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterAWApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   PutInteger(token, keyRadius, m_sData->nRadius);
   PutFloat(token, keyNoise, m_sData->dNoise);
   PutBoolean(token, keyAuto, m_sData->bAuto);
   if(m_sData->bKeepNoise)
      PutBoolean(token, keyKeepNoise, m_sData->bKeepNoise);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterAWApp object

CFilterAWApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
