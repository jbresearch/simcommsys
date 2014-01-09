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
#include "FilterExport.h"
#include "FilterExportDlg.h"
#include "ScriptingKeys.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterExportApp

BEGIN_MESSAGE_MAP(CFilterExportApp, CWinApp)
//{{AFX_MSG_MAP(CFilterExportApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterExportApp construction

CFilterExportApp::CFilterExportApp() : CPSPlugIn(sizeof(SFilterExportData), 100)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterExportApp filter selector functions

// show the about dialog here
void CFilterExportApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterExportApp::FilterStart(void)
   {
   // tile the image row by row, keeping the same tile area as suggested
   SetTileWidth(GetImageWidth());
   SetTileHeight(max(1,GetSuggestedTileHeight()*GetSuggestedTileWidth()/GetImageWidth()));
   // show dialog if necessary, set up first tile, start progress indicator & timer
   CPSPlugIn::FilterStart();
   // open file for writing
   file.open(m_sData->sPathName);
   }

void CFilterExportApp::FilterContinue(void)
   {
   // update progress counter
   DisplayTileProgress(0);

   // get this tile for processing
   libbase::matrix<double> in;
   GetPixelMatrix(in);

   // save this tile as a matrix (data only)
   in.serialize(file);

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   }

void CFilterExportApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();
   // close file
   file.close();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterExportApp helper functions

void CFilterExportApp::ShowDialog(void)
   {
   CFileDialog dlg(FALSE, "*.dat", m_sData->sPathName);

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   strcpy(m_sData->sPathName, dlg.GetPathName());

   SetShowDialog(false);
   }

void CFilterExportApp::InitPointer(char* sData)
   {
   m_sData = (SFilterExportData *) sData;
   }

void CFilterExportApp::InitParameters()
   {
   strcpy(m_sData->sPathName, "*.*");
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterExportApp scripting support

void CFilterExportApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      case keyFileName:
         GetString(token, m_sData->sPathName);
         break;
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterExportApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   PutString(token, keyFileName, m_sData->sPathName);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterExportApp object

CFilterExportApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
