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
#include "FilterOrphans.h"
#include "FilterOrphansDlg.h"
#include "ScriptingKeys.h"
#include <math.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansApp

BEGIN_MESSAGE_MAP(CFilterOrphansApp, CWinApp)
//{{AFX_MSG_MAP(CFilterOrphansApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansApp construction

CFilterOrphansApp::CFilterOrphansApp() : CPSPlugIn(sizeof(SFilterOrphansData), 101)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansApp filter selector functions

// show the about dialog here
void CFilterOrphansApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterOrphansApp::FilterStart(void)
   {
   // FilterStart will get user parameters if necessary & select the first tile
   CPSPlugIn::FilterStart();
   // Once we have radius, setup overlap for future tile updates
   SetTileOverlap(2);
   }

void CFilterOrphansApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0);

   // do the processing for this tile
   libbase::matrix<double> in, out;
   GetPixelMatrix(in);
   out = in;
   for(int i=1; i<in.size().rows()-1; i++)
      for(int j=1; j<in.size().cols()-1; j++)
         if(in(i,j) < 0.5) // current pixel is black
            {
            int sum = -1;  // to discount current pixel
            for(int ii=-1; ii<=1; ii++)
               for(int jj=-1; jj<=1; jj++)
                  if(in(i+ii, j+jj) < 0.5)
                     sum++;
            out(i,j) = (sum < m_sData->nWeight) ? 1 : 0;
            }
         else
            out(i,j) = 1;
   SetPixelMatrix(out);

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   }

void CFilterOrphansApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansApp helper functions

void CFilterOrphansApp::ShowDialog(void)
   {
   CFilterOrphansDlg   dlg;

   dlg.m_pPSPlugIn = this;

   dlg.m_bKeepNoise = m_sData->bKeepNoise;
   dlg.m_nWeight = m_sData->nWeight;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   m_sData->bKeepNoise = dlg.m_bKeepNoise != 0;
   m_sData->nWeight = dlg.m_nWeight;
   SetShowDialog(false);
   }

void CFilterOrphansApp::InitPointer(char* sData)
   {
   m_sData = (SFilterOrphansData *) sData;
   }

void CFilterOrphansApp::InitParameters()
   {
   m_sData->bKeepNoise = false;
   m_sData->nWeight = 1;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansApp scripting support

void CFilterOrphansApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      case keyWeight:
         GetInteger(token, &m_sData->nWeight);
         break;
      case keyKeepNoise:
         GetBoolean(token, &m_sData->bKeepNoise);
         break;
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterOrphansApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   PutInteger(token, keyWeight, m_sData->nWeight);
   if(m_sData->bKeepNoise)
      PutBoolean(token, keyKeepNoise, m_sData->bKeepNoise);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterOrphansApp object

CFilterOrphansApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
