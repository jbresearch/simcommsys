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
#include "AutomateEmbedding.h"
#include "AutomateEmbeddingDlg.h"
#include "ScriptingKeys.h"

#include <fstream>
#include <string>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingApp

BEGIN_MESSAGE_MAP(CAutomateEmbeddingApp, CWinApp)
//{{AFX_MSG_MAP(CAutomateEmbeddingApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingApp construction/destruction

CAutomateEmbeddingApp::CAutomateEmbeddingApp()
   {
   m_sData = new SAutomateEmbeddingData;
   }

CAutomateEmbeddingApp::~CAutomateEmbeddingApp()
   {
   delete m_sData;
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingApp filter selector functions

// show the about dialog here
void CAutomateEmbeddingApp::About(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CAutomateEmbeddingApp::Process(void)
   {
   // prepare variables
   CString sTempName;
   // main processing loop
   libbase::timer tDuration;
   for(double dStrength = m_sData->dStrengthMin; dStrength <= m_sData->dStrengthMax; dStrength += m_sData->dStrengthStep)
      {
      PlayeventFilterEmbed(/*embedding*/ 0, 1, dStrength, \
                           /*interleaver*/ false, 0, 1, \
                           /*source*/ 2, 1, "", \
                           /*codec*/ "", "");
      if(m_sData->bJpeg)
         {
         for(int nJpegQ = m_sData->nJpegMin; nJpegQ <= m_sData->nJpegMax; nJpegQ += m_sData->nJpegStep)
            {
            // save as TIFF/JPEG
            sTempName.Format("%s\\es%2.1f-q%02d.tif", m_sData->sOutput, dStrength, nJpegQ);
            PlayeventSaveJPEG(nJpegQ, sTempName, true);
            }
         }
      else
         {
         // save as TIFF/LZW
         sTempName.Format("%s\\es%2.1f.tif", m_sData->sOutput, dStrength);
         PlayeventSaveLZW(sTempName, true);
         }
      PlayeventSelectState(-1);
      }
   // tell the user how long it took
   tDuration.stop();
   CString sTemp;
   sTemp.Format("Time taken: %s", std::string(tDuration).c_str());
   MessageBox(NULL, sTemp, "Automate Embedding", MB_OK);
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingApp virtual overrides

void CAutomateEmbeddingApp::ShowDialog(void)
   {
   CAutomateEmbeddingDlg   dlg;

   dlg.m_pPSAutomate = this;

   // path for output files
   dlg.m_sOutput = m_sData->sOutput;
   // system options
   dlg.m_bJpeg = m_sData->bJpeg;
   // range of embedding strengths
   dlg.m_dStrengthMin = m_sData->dStrengthMin;
   dlg.m_dStrengthMax = m_sData->dStrengthMax;
   dlg.m_dStrengthStep = m_sData->dStrengthStep;
   // range of JPEG compression quality (if requested)
   dlg.m_nJpegMin = m_sData->nJpegMin;
   dlg.m_nJpegMax = m_sData->nJpegMax;
   dlg.m_nJpegStep = m_sData->nJpegStep;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((SPErr)userCanceledErr);

   // path for output files
   strcpy(m_sData->sOutput, dlg.m_sOutput);
   // system options
   m_sData->bJpeg = dlg.m_bJpeg != 0;
   // range of embedding strengths
   m_sData->dStrengthMin = dlg.m_dStrengthMin;
   m_sData->dStrengthMax = dlg.m_dStrengthMax;
   m_sData->dStrengthStep = dlg.m_dStrengthStep;
   // range of JPEG compression quality (if requested)
   m_sData->nJpegMin = dlg.m_nJpegMin;
   m_sData->nJpegMax = dlg.m_nJpegMax;
   m_sData->nJpegStep = dlg.m_nJpegStep;
   }

void CAutomateEmbeddingApp::InitParameters()
   {
   // path for output files
   strcpy(m_sData->sOutput, "");
   // system options
   m_sData->bJpeg = false;
   // range of embedding strengths
   m_sData->dStrengthMin = -48;
   m_sData->dStrengthMax = -26;
   m_sData->dStrengthStep = 1;
   // range of JPEG compression quality (if requested)
   m_sData->bJpeg = false;
   m_sData->nJpegMin = 0;
   m_sData->nJpegMax = 12;
   m_sData->nJpegStep = 1;
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingApp scripting support

void CAutomateEmbeddingApp::ReadScriptParameters(PIActionDescriptor descriptor)
   {
   // path for output files
   GetString(descriptor, keyOutput, m_sData->sOutput);
   // system options
   GetBoolean(descriptor, keyJpeg, &m_sData->bJpeg);
   // range of embedding strengths
   GetFloat(descriptor, keyStrengthMin, &m_sData->dStrengthMin);
   GetFloat(descriptor, keyStrengthMax, &m_sData->dStrengthMax);
   GetFloat(descriptor, keyStrengthStep, &m_sData->dStrengthStep);
   // range of JPEG compression quality (if requested)
   GetInteger(descriptor, keyJpegMin, &m_sData->nJpegMin);
   GetInteger(descriptor, keyJpegMax, &m_sData->nJpegMax);
   GetInteger(descriptor, keyJpegStep, &m_sData->nJpegStep);
   }

void CAutomateEmbeddingApp::WriteScriptParameters(PIActionDescriptor descriptor)
   {
   // path for output files
   PutString(descriptor, keyOutput, m_sData->sOutput);
   // system options
   if(m_sData->bJpeg)
      PutBoolean(descriptor, keyJpeg, m_sData->bJpeg);
   // range of embedding strengths
   PutFloat(descriptor, keyStrengthMin, m_sData->dStrengthMin);
   PutFloat(descriptor, keyStrengthMax, m_sData->dStrengthMax);
   PutFloat(descriptor, keyStrengthStep, m_sData->dStrengthStep);
   // range of JPEG compression quality (if requested)
   if(m_sData->bJpeg)
      {
      PutInteger(descriptor, keyJpegMin, m_sData->nJpegMin);
      PutInteger(descriptor, keyJpegMax, m_sData->nJpegMax);
      PutInteger(descriptor, keyJpegStep, m_sData->nJpegStep);
      }
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CAutomateEmbeddingApp object

CAutomateEmbeddingApp theApp;

DLLExport SPAPI SPErr PluginMain(const char* sCaller, const char* sSelector, void* pData)
   {
   return theApp.Main(sCaller, sSelector, pData);
   }
