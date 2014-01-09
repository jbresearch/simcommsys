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
#include "AutomateGraphing.h"
#include "AutomateGraphingDlg.h"
#include "ScriptingKeys.h"

#include <fstream>
#include <string>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp

BEGIN_MESSAGE_MAP(CAutomateGraphingApp, CWinApp)
//{{AFX_MSG_MAP(CAutomateGraphingApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp construction/destruction

CAutomateGraphingApp::CAutomateGraphingApp()
   {
   m_sData = new SAutomateGraphingData;
   }

CAutomateGraphingApp::~CAutomateGraphingApp()
   {
   delete m_sData;
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp filter selector functions

// show the about dialog here
void CAutomateGraphingApp::About(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CAutomateGraphingApp::Process(void)
   {
   // prepare variables
   CString sTempName;
   if(m_sData->bJpeg)
      sTempName.Format("%s.tif", _tempnam("c:\\tmp", "steg"));
   // read parameters
   ReadParameters();
   // write file header
   WriteHeader();
   // main processing loop
   libbase::timer tDuration;
   for(double dStrength = m_sData->dStrengthMin; dStrength <= m_sData->dStrengthMax; dStrength += m_sData->dStrengthStep)
      {
      PlayeventFilterEmbed(/*embedding*/ 0, 1, dStrength, \
                           /*interleaver*/ true, 0, 1, \
                           /*source*/ 2, 1, "", \
                           /*codec*/ "", "");
      if(m_sData->bJpeg)
         {
         for(int nJpegQ = m_sData->nJpegMin; nJpegQ <= m_sData->nJpegMax; nJpegQ += m_sData->nJpegStep)
            {
            // write row header
            std::ofstream file(m_sData->sResults, std::ios::out | std::ios::app);
            file << dStrength << "\t" << nJpegQ << "\t";
            file.close();
            // simulate JPEG channel
            PlayeventSaveJPEG(nJpegQ, sTempName, true);
            PlayeventOpen(sTempName);
            remove(sTempName);
            // main processing
            DoExtract(dStrength);
            PlayeventClose(false);
            }
         }
      else
         {
         // write row header
         std::ofstream file(m_sData->sResults, std::ios::out | std::ios::app);
         file << dStrength << "\t";
         file.close();
         // main processing
         DoExtract(dStrength);
         PlayeventSelectState(-3);
         }
      PlayeventSelectState(-1);
      }
   // tell the user how long it took
   tDuration.stop();
   CString sTemp;
   sTemp.Format("Time taken: %s", std::string(tDuration).c_str());
   MessageBox(NULL, sTemp, "Automate Graphing", MB_OK);
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp helper functions

void CAutomateGraphingApp::ReadParameters()
   {
   std::ifstream file(m_sData->sParameters);
   std::string s;
   getline(file, s);
   m_sParameters.nFilterType = atoi(s.c_str());
   switch(m_sParameters.nFilterType)
      {
      case 0:
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterATM.nRadius = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterATM.nAlpha = atoi(s.c_str());
         break;
      case 1:
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterAW.nRadius = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterAW.dNoise = atof(s.c_str());
         break;
      case 2:
         // wavelet basis
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nWaveletType = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nWaveletPar = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nWaveletLevel = atoi(s.c_str());
         // thresholding
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nThreshType = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nThreshSelector = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.dThreshCutoff = atof(s.c_str());
         // tiling
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nTileX = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.nTileY = atoi(s.c_str());
         getline(file, s);
         m_sParameters.uFilterSettings.sFilterWavelet.bWholeImage = atoi(s.c_str()) != 0;
         break;
      default:
         throw((SPErr)kSPLogicError);
         break;
      }
   }

void CAutomateGraphingApp::WriteHeader()
   {
   std::ofstream file(m_sData->sResults);
   file << "# Strength";
   if(m_sData->bJpeg)
      file << "\tQuality";
   if(m_sData->bPrintBER)
      file << "\tBER";
   if(m_sData->bPrintSNR)
      file << "\tSNR";
   if(m_sData->bPrintEstimate)
      file << "\tEstimate";
   if(m_sData->bPrintChiSquare)
      file << "\tChiSqr";
   file << "\n";
   file.close();
   }

void CAutomateGraphingApp::DoExtract(double dStrength)
   {
   PlayeventConvertMode(16);
   switch(m_sParameters.nFilterType)
      {
      case 0:
         PlayeventFilterATM( \
            m_sParameters.uFilterSettings.sFilterATM.nRadius, \
            m_sParameters.uFilterSettings.sFilterATM.nAlpha, \
            true);
         break;
      case 1:
         PlayeventFilterAW( \
            m_sParameters.uFilterSettings.sFilterAW.nRadius, \
            m_sParameters.uFilterSettings.sFilterAW.dNoise, \
            true);
         break;
      case 2:
         PlayeventFilterWavelet( \
            m_sParameters.uFilterSettings.sFilterWavelet.nWaveletType, \
            m_sParameters.uFilterSettings.sFilterWavelet.nWaveletPar, \
            m_sParameters.uFilterSettings.sFilterWavelet.nWaveletLevel, \
            m_sParameters.uFilterSettings.sFilterWavelet.nThreshType, \
            m_sParameters.uFilterSettings.sFilterWavelet.nThreshSelector, \
            m_sParameters.uFilterSettings.sFilterWavelet.dThreshCutoff, \
            m_sParameters.uFilterSettings.sFilterWavelet.nTileX, \
            m_sParameters.uFilterSettings.sFilterWavelet.nTileY, \
            m_sParameters.uFilterSettings.sFilterWavelet.bWholeImage, \
            true);
         break;
      }
   PlayeventFilterExtract(/*embedding*/ 0, 1, dStrength, m_sData->bPresetStrength, \
                          /*interleaver*/ true, 0, 1, \
                          /*source*/ 2, 1, "", \
                          /*codec*/ "", "", \
                          /*results*/ m_sData->sResults, "", "", "", "", \
                          /*compute*/ m_sData->bPrintBER, m_sData->bPrintSNR, m_sData->bPrintEstimate, m_sData->bPrintChiSquare, 0);
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp virtual overrides

void CAutomateGraphingApp::ShowDialog(void)
   {
   CAutomateGraphingDlg   dlg;

   dlg.m_pPSAutomate = this;

   // files with system parameters (input) and results (output)
   dlg.m_sParameters = m_sData->sParameters;
   dlg.m_sResults = m_sData->sResults;
   // system options
   dlg.m_bJpeg = m_sData->bJpeg;
   dlg.m_bPresetStrength = m_sData->bPresetStrength;
   // range of embedding strengths
   dlg.m_dStrengthMin = m_sData->dStrengthMin;
   dlg.m_dStrengthMax = m_sData->dStrengthMax;
   dlg.m_dStrengthStep = m_sData->dStrengthStep;
   // range of JPEG compression quality (if requested)
   dlg.m_nJpegMin = m_sData->nJpegMin;
   dlg.m_nJpegMax = m_sData->nJpegMax;
   dlg.m_nJpegStep = m_sData->nJpegStep;
   // requested outputs
   dlg.m_bPrintBER = m_sData->bPrintBER;
   dlg.m_bPrintSNR = m_sData->bPrintSNR;
   dlg.m_bPrintEstimate = m_sData->bPrintEstimate;
   dlg.m_bPrintChiSquare = m_sData->bPrintChiSquare;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((SPErr)userCanceledErr);

   // files with system parameters (input) and results (output)
   strcpy(m_sData->sParameters, dlg.m_sParameters);
   strcpy(m_sData->sResults, dlg.m_sResults);
   // system options
   m_sData->bJpeg = dlg.m_bJpeg != 0;
   m_sData->bPresetStrength = dlg.m_bPresetStrength != 0;
   // range of embedding strengths
   m_sData->dStrengthMin = dlg.m_dStrengthMin;
   m_sData->dStrengthMax = dlg.m_dStrengthMax;
   m_sData->dStrengthStep = dlg.m_dStrengthStep;
   // range of JPEG compression quality (if requested)
   m_sData->nJpegMin = dlg.m_nJpegMin;
   m_sData->nJpegMax = dlg.m_nJpegMax;
   m_sData->nJpegStep = dlg.m_nJpegStep;
   // requested outputs
   m_sData->bPrintBER = dlg.m_bPrintBER != 0;
   m_sData->bPrintSNR = dlg.m_bPrintSNR != 0;
   m_sData->bPrintEstimate = dlg.m_bPrintEstimate != 0;
   m_sData->bPrintChiSquare = dlg.m_bPrintChiSquare != 0;
   }

void CAutomateGraphingApp::InitParameters()
   {
   // files with system parameters (input) and results (output)
   strcpy(m_sData->sParameters, "");
   strcpy(m_sData->sResults, "");
   // system options
   m_sData->bJpeg = false;
   m_sData->bPresetStrength = false;
   // range of embedding strengths
   m_sData->dStrengthMin = -48;
   m_sData->dStrengthMax = -26;
   m_sData->dStrengthStep = 1;
   // range of JPEG compression quality (if requested)
   m_sData->bJpeg = false;
   m_sData->nJpegMin = 0;
   m_sData->nJpegMax = 12;
   m_sData->nJpegStep = 1;
   // requested outputs
   m_sData->bPrintBER = true;
   m_sData->bPrintSNR = true;
   m_sData->bPrintEstimate = true;
   m_sData->bPrintChiSquare = true;
   }

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingApp scripting support

void CAutomateGraphingApp::ReadScriptParameters(PIActionDescriptor descriptor)
   {
   // files with system parameters (input) and results (output)
   GetString(descriptor, keyParameters, m_sData->sParameters);
   GetString(descriptor, keyResults, m_sData->sResults);
   // system options
   GetBoolean(descriptor, keyJpeg, &m_sData->bJpeg);
   GetBoolean(descriptor, keyPresetStrength, &m_sData->bPresetStrength);
   // range of embedding strengths
   GetFloat(descriptor, keyStrengthMin, &m_sData->dStrengthMin);
   GetFloat(descriptor, keyStrengthMax, &m_sData->dStrengthMax);
   GetFloat(descriptor, keyStrengthStep, &m_sData->dStrengthStep);
   // range of JPEG compression quality (if requested)
   GetInteger(descriptor, keyJpegMin, &m_sData->nJpegMin);
   GetInteger(descriptor, keyJpegMax, &m_sData->nJpegMax);
   GetInteger(descriptor, keyJpegStep, &m_sData->nJpegStep);
   // requested outputs
   GetBoolean(descriptor, keyPrintBER, &m_sData->bPrintBER);
   GetBoolean(descriptor, keyPrintSNR, &m_sData->bPrintSNR);
   GetBoolean(descriptor, keyPrintEstimate, &m_sData->bPrintEstimate);
   GetBoolean(descriptor, keyPrintChiSquare, &m_sData->bPrintChiSquare);
   }

void CAutomateGraphingApp::WriteScriptParameters(PIActionDescriptor descriptor)
   {
   // files with system parameters (input) and results (output)
   PutString(descriptor, keyParameters, m_sData->sParameters);
   PutString(descriptor, keyResults, m_sData->sResults);
   // system options
   if(m_sData->bJpeg)
      PutBoolean(descriptor, keyJpeg, m_sData->bJpeg);
   if(m_sData->bPresetStrength)
      PutBoolean(descriptor, keyPresetStrength, m_sData->bPresetStrength);
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
   // requested outputs
   PutBoolean(descriptor, keyPrintBER, m_sData->bPrintBER);
   PutBoolean(descriptor, keyPrintSNR, m_sData->bPrintSNR);
   PutBoolean(descriptor, keyPrintEstimate, m_sData->bPrintEstimate);
   PutBoolean(descriptor, keyPrintChiSquare, m_sData->bPrintChiSquare);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CAutomateGraphingApp object

CAutomateGraphingApp theApp;

DLLExport SPAPI SPErr PluginMain(const char* sCaller, const char* sSelector, void* pData)
   {
   return theApp.Main(sCaller, sSelector, pData);
   }
