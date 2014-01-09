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
#include "FilterVariance.h"
#include "FilterVarianceDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "variancefilter.h"
#include "limiter.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceApp

BEGIN_MESSAGE_MAP(CFilterVarianceApp, CWinApp)
//{{AFX_MSG_MAP(CFilterVarianceApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceApp construction

CFilterVarianceApp::CFilterVarianceApp() : CPSPlugIn(sizeof(SFilterVarianceData), 101)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceApp filter selector functions

// show the about dialog here
void CFilterVarianceApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterVarianceApp::FilterStart(void)
   {
   // FilterStart will get user parameters if necessary & select the first tile
   CPSPlugIn::FilterStart();
   // Once we have radius, setup overlap for future tile updates (this is not
   // needed for the first tile anyway.
   SetTileOverlap(2 * m_sData->nRadius);
   // initialize variables used for scaling
   m_nIteration = 0;
   m_dScale = 0;
   }

void CFilterVarianceApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0, 100, m_nIteration, m_sData->nScale==0 ? 2 : 1);

   // get this tile for processing
   libbase::matrix<double> in, out;
   GetPixelMatrix(in);

   // get variance for each pixel
   libimage::variancefilter<double> filter(m_sData->nRadius);
   filter.process(in, out);

   switch(m_nIteration)
      {
      case 0:
         // if we don't want auto-scaling, scale this tile & write back
         if(m_sData->nScale != 0)
            {
            out *= m_sData->nScale;
            libimage::limiter<double> lim(0,1);
            lim.process(out);
            SetPixelMatrix(out);
            }
         else
            {
            // otherwise, to be able to normalize later, update scale info
            const double dScale = out.max();
            if(m_dScale < dScale)
               m_dScale = dScale;
            }
         break;

      case 1:
         // normalize based on scale found earlier & write back
         out /= m_dScale;
         libimage::limiter<double> lim(0,1);
         lim.process(out);
         SetPixelMatrix(out);
         break;
      }

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   // if we need to do a second iteration
   if(m_nIteration==0 && m_sData->nScale==0 && IterationDone())
      {
      m_nIteration++;
      IterationStart();
      }
   }

void CFilterVarianceApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceApp helper functions

void CFilterVarianceApp::ShowDialog(void)
   {
   CFilterVarianceDlg   dlg;

   dlg.m_pPSPlugIn = this;

   dlg.m_nRadius = m_sData->nRadius;
   dlg.m_nScale = m_sData->nScale;
   dlg.m_bAutoScale = (m_sData->nScale == 0);

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   m_sData->nRadius = dlg.m_nRadius;
   m_sData->nScale = dlg.m_bAutoScale ? 0 : dlg.m_nScale;
   SetShowDialog(false);
   }

void CFilterVarianceApp::InitPointer(char* sData)
   {
   m_sData = (SFilterVarianceData *) sData;
   }

void CFilterVarianceApp::InitParameters()
   {
   m_sData->nRadius = 1;
   m_sData->nScale = 1;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceApp scripting support

void CFilterVarianceApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      case keyRadius:
         GetInteger(token, &m_sData->nRadius);
         break;
      case keyScale:
         GetInteger(token, &m_sData->nScale);
         break;
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterVarianceApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   PutInteger(token, keyRadius, m_sData->nRadius);
   PutInteger(token, keyScale, m_sData->nScale);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterVarianceApp object

CFilterVarianceApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
