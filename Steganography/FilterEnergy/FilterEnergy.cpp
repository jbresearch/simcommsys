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
#include "FilterEnergy.h"
#include "FilterEnergyDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyApp

BEGIN_MESSAGE_MAP(CFilterEnergyApp, CWinApp)
//{{AFX_MSG_MAP(CFilterEnergyApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyApp construction

CFilterEnergyApp::CFilterEnergyApp() : CPSPlugIn(sizeof(SFilterEnergyData), 101)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyApp filter selector functions

// show the about dialog here
void CFilterEnergyApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterEnergyApp::FilterStart(void)
   {
   // show dialog if necessary, set up first tile, start progress indicator & timer
   CPSPlugIn::FilterStart();
   // reset the statistics
   rv.reset();
   }

void CFilterEnergyApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0);

   // get this tile for processing
   libbase::matrix<double> in;
   GetPixelMatrix(in);

   // update the statistics
   rv.insert(in);

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   }

void CFilterEnergyApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   // prepare result
   CString sTemp, sDisplay;
   sDisplay = "";
   if(m_sData->bDisplayEnergy)
      {
      sTemp.Format("%0.2f", 10*log10(rv.var()));
      sDisplay += sTemp;
      }
   if(m_sData->bDisplayVariance)
      {
      sTemp.Format("%0.6f", rv.var());
      if(m_sData->bDisplayEnergy)
         sDisplay += '\t';
      sDisplay += sTemp;
      }
   if(m_sData->bDisplayPixelCount)
      {
      sTemp.Format("%d", rv.count());
      if(m_sData->bDisplayEnergy || m_sData->bDisplayVariance)
         sDisplay += '\t';
      sDisplay += sTemp;
      }
   // display or add to file, as requested
   if(strlen(m_sData->sFileName) > 0)
      {
      std::ofstream file(m_sData->sFileName, std::ios::out | (m_sData->bAppend ? std::ios::app : std::ios::trunc));
      file << LPCTSTR(sDisplay) << "\n";
      }
   else
      AfxMessageBox(sDisplay);
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyApp helper functions

void CFilterEnergyApp::ShowDialog(void)
   {
   CFilterEnergyDlg   dlg;

   dlg.m_pPSPlugIn = this;

   dlg.m_bScreenOnly = (strlen(m_sData->sFileName) == 0);
   dlg.m_sFileName = m_sData->sFileName;
   dlg.m_bAppend = m_sData->bAppend;
   dlg.m_bDisplayVariance = m_sData->bDisplayVariance;
   dlg.m_bDisplayEnergy = m_sData->bDisplayEnergy;
   dlg.m_bDisplayPixelCount = m_sData->bDisplayPixelCount;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   strcpy(m_sData->sFileName, dlg.m_bScreenOnly ? "" : dlg.m_sFileName);
   m_sData->bAppend = dlg.m_bAppend!=0;
   m_sData->bDisplayVariance = dlg.m_bDisplayVariance!=0;
   m_sData->bDisplayEnergy = dlg.m_bDisplayEnergy!=0;
   m_sData->bDisplayPixelCount = dlg.m_bDisplayPixelCount!=0;
   SetShowDialog(false);
   }

void CFilterEnergyApp::InitPointer(char* sData)
   {
   m_sData = (SFilterEnergyData *) sData;
   }

void CFilterEnergyApp::InitParameters()
   {
   strcpy(m_sData->sFileName, "");
   m_sData->bAppend = false;
   m_sData->bDisplayVariance = false;
   m_sData->bDisplayEnergy = true;
   m_sData->bDisplayPixelCount = false;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyApp scripting support

void CFilterEnergyApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      case keyFileName:
         GetString(token, m_sData->sFileName);
         break;
      case keyAppend:
         GetBoolean(token, &m_sData->bAppend);
         break;
      case keyDisplayVariance:
         GetBoolean(token, &m_sData->bDisplayVariance);
         break;
      case keyDisplayEnergy:
         GetBoolean(token, &m_sData->bDisplayEnergy);
         break;
      case keyDisplayPixelCount:
         GetBoolean(token, &m_sData->bDisplayPixelCount);
         break;
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterEnergyApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   PutString(token, keyFileName, m_sData->sFileName);
   if(m_sData->bAppend)
      PutBoolean(token, keyAppend, m_sData->bAppend);
   if(m_sData->bDisplayVariance)
      PutBoolean(token, keyDisplayVariance, m_sData->bDisplayVariance);
   if(!m_sData->bDisplayEnergy)
      PutBoolean(token, keyDisplayEnergy, m_sData->bDisplayEnergy);
   if(m_sData->bDisplayPixelCount)
      PutBoolean(token, keyDisplayPixelCount, m_sData->bDisplayPixelCount);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterEnergyApp object

CFilterEnergyApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
