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
#include "FilterExtract.h"
#include "FilterExtractDlg.h"
#include "ComputeStrengthDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
   {
   //{{AFX_DATA_INIT(CAboutDlg)
   //}}AFX_DATA_INIT
   }

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CAboutDlg)
   //}}AFX_DATA_MAP
   }

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
//{{AFX_MSG_MAP(CAboutDlg)
// No message handlers
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractDlg dialog

CFilterExtractDlg::CFilterExtractDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterExtractDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterExtractDlg)
        m_sPuncture = _T("");
        m_sCodec = _T("");
        m_bInterleave = FALSE;
        m_bPresetStrength = FALSE;
        m_nEmbedRate = 0;
        m_nEmbedSeed = 0;
        m_dEmbedStrength = 0.0;
        m_sEmbedded = _T("");
        m_sExtracted = _T("");
        m_nInterleaverSeed = 0;
        m_dInterleaverDensity = 0.0;
        m_sSource = _T("");
        m_nSourceSeed = 0;
        m_nSourceType = -1;
        m_sUniform = _T("");
        m_sDecoded = _T("");
        m_sResults = _T("");
        m_nFeedback = -1;
        m_bPrintBER = FALSE;
        m_bPrintSNR = FALSE;
        m_bPrintEstimate = FALSE;
        m_bPrintChiSquare = FALSE;
        m_sEmbeddedImage = _T("");
        m_sExtractedImage = _T("");
        //}}AFX_DATA_INIT
   }


void CFilterExtractDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterExtractDlg)
        DDX_Text(pDX, IDC_PUNCTURE, m_sPuncture);
        DDX_Text(pDX, IDC_CODEC, m_sCodec);
        DDX_Check(pDX, IDC_INTERLEAVE, m_bInterleave);
        DDX_Check(pDX, IDC_PRESET_STRENGTH, m_bPresetStrength);
        DDX_Text(pDX, IDC_EMBED_RATE, m_nEmbedRate);
        DDX_Text(pDX, IDC_EMBED_SEED, m_nEmbedSeed);
        DDX_Text(pDX, IDC_EMBED_STRENGTH, m_dEmbedStrength);
        DDX_Text(pDX, IDC_EMBEDDED, m_sEmbedded);
        DDX_Text(pDX, IDC_EXTRACTED, m_sExtracted);
        DDX_Text(pDX, IDC_INTERLEAVER_SEED, m_nInterleaverSeed);
        DDX_Text(pDX, IDC_INTERLEAVER_DENSITY, m_dInterleaverDensity);
        DDX_Text(pDX, IDC_SOURCE, m_sSource);
        DDX_Text(pDX, IDC_SOURCE_SEED, m_nSourceSeed);
        DDX_CBIndex(pDX, IDC_SOURCE_TYPE, m_nSourceType);
        DDX_Text(pDX, IDC_UNIFORM, m_sUniform);
        DDX_Text(pDX, IDC_DECODED, m_sDecoded);
        DDX_Text(pDX, IDC_RESULTS, m_sResults);
        DDX_CBIndex(pDX, IDC_FEEDBACK, m_nFeedback);
        DDX_Check(pDX, IDC_PRINT_BER, m_bPrintBER);
        DDX_Check(pDX, IDC_PRINT_SNR, m_bPrintSNR);
        DDX_Check(pDX, IDC_PRINT_ESTIMATE, m_bPrintEstimate);
        DDX_Check(pDX, IDC_PRINT_CHI_SQUARE, m_bPrintChiSquare);
        DDX_Text(pDX, IDC_EMBEDDED_IMAGE, m_sEmbeddedImage);
        DDX_Text(pDX, IDC_EXTRACTED_IMAGE, m_sExtractedImage);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterExtractDlg, CDialog)
//{{AFX_MSG_MAP(CFilterExtractDlg)
        ON_BN_CLICKED(IDC_LOAD_SOURCE, OnLoadSource)
        ON_BN_CLICKED(IDC_LOAD_CODEC, OnLoadCodec)
        ON_BN_CLICKED(IDC_LOAD_PUNCTURE, OnLoadPuncture)
        ON_BN_CLICKED(IDC_CLEAR_SOURCE, OnClearSource)
        ON_BN_CLICKED(IDC_CLEAR_CODEC, OnClearCodec)
        ON_BN_CLICKED(IDC_CLEAR_PUNCTURE, OnClearPuncture)
        ON_BN_CLICKED(IDC_COMPUTE_STRENGTH, OnComputeStrength)
        ON_BN_CLICKED(IDC_INTERLEAVE, OnInterleave)
        ON_CBN_SELCHANGE(IDC_SOURCE_TYPE, OnSelchangeSourceType)
        ON_EN_CHANGE(IDC_INTERLEAVER_DENSITY, OnChangeInterleaverDensity)
        ON_EN_CHANGE(IDC_EMBED_RATE, OnChangeEmbedRate)
        ON_BN_CLICKED(IDC_PRESET_STRENGTH, OnPresetStrength)
        ON_BN_CLICKED(IDC_SAVE_EMBEDDED, OnSaveEmbedded)
        ON_BN_CLICKED(IDC_SAVE_EXTRACTED, OnSaveExtracted)
        ON_BN_CLICKED(IDC_SAVE_UNIFORM, OnSaveUniform)
        ON_BN_CLICKED(IDC_CLEAR_EMBEDDED, OnClearEmbedded)
        ON_BN_CLICKED(IDC_CLEAR_EXTRACTED, OnClearExtracted)
        ON_BN_CLICKED(IDC_CLEAR_UNIFORM, OnClearUniform)
        ON_BN_CLICKED(IDC_CLEAR_DECODED, OnClearDecoded)
        ON_BN_CLICKED(IDC_SAVE_DECODED, OnSaveDecoded)
        ON_BN_CLICKED(IDC_CLEAR_RESULTS, OnClearResults)
        ON_BN_CLICKED(IDC_SAVE_RESULTS, OnSaveResults)
        ON_BN_CLICKED(IDC_CLEAR_EMBEDDED_IMAGE, OnClearEmbeddedImage)
        ON_BN_CLICKED(IDC_SAVE_EMBEDDED_IMAGE, OnSaveEmbeddedImage)
        ON_BN_CLICKED(IDC_CLEAR_EXTRACTED_IMAGE, OnClearExtractedImage)
        ON_BN_CLICKED(IDC_SAVE_EXTRACTED_IMAGE, OnSaveExtractedImage)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractDlg message handlers

BOOL CFilterExtractDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   m_nRawSize = m_pPSPlugIn->GetImageWidth() * m_pPSPlugIn->GetImageHeight() * m_pPSPlugIn->GetPlanes();
   ComputeFileData();
   ComputeCodecData();
   ComputePunctureData();
   UpdateDisplay();

   GetDlgItem(IDC_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_CLEAR_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_LOAD_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_SOURCE_SEED)->EnableWindow(m_nSourceType==2);

   GetDlgItem(IDC_INTERLEAVER_DENSITY)->EnableWindow(m_bInterleave);
   GetDlgItem(IDC_INTERLEAVER_SEED)->EnableWindow(m_bInterleave);

   GetDlgItem(IDC_EMBED_STRENGTH)->EnableWindow(m_bPresetStrength);
   GetDlgItem(IDC_COMPUTE_STRENGTH)->EnableWindow(m_bPresetStrength);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterExtractDlg::OnOK()
   {
   UpdateData(true);
   if(m_nSourceType == 3 && m_sSource.IsEmpty())
      {
      MessageBox("Missing source filename.", NULL, MB_OK | MB_ICONWARNING);
      return;
      }
   if(m_nEmbedRate < 1)
      {
      MessageBox("Invalid bandwidth expansion rate.", NULL, MB_OK | MB_ICONWARNING);
      return;
      }
   if(m_bInterleave && (m_dInterleaverDensity <= 0 || m_dInterleaverDensity > 1))
      {
      MessageBox("Density must be between 0 and 1.", NULL, MB_OK | MB_ICONWARNING);
      return;
      }

   CDialog::OnOK();
   }

// message events

void CFilterExtractDlg::OnChangeEmbedRate()
   {
   m_nEmbedRate = GetDlgItemInt(IDC_EMBED_RATE);
   if(m_nEmbedRate < 1)
      m_nEmbedRate = 1;
   UpdateDisplay();
   }

void CFilterExtractDlg::OnChangeInterleaverDensity()
   {
   CString sTemp;
   GetDlgItemText(IDC_INTERLEAVER_DENSITY, sTemp);
   m_dInterleaverDensity = atof(sTemp);
   UpdateDisplay();
   }

void CFilterExtractDlg::OnSelchangeSourceType()
   {
   UpdateData(true);
   //m_nSourceType = GetDlgItemInt(IDC_SOURCE_TYPE);
   GetDlgItem(IDC_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_CLEAR_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_LOAD_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_SOURCE_SEED)->EnableWindow(m_nSourceType==2);
   UpdateDisplay();
   }

void CFilterExtractDlg::OnInterleave()
   {
   m_bInterleave = ((CButton*)GetDlgItem(IDC_INTERLEAVE))->GetCheck();
   GetDlgItem(IDC_INTERLEAVER_DENSITY)->EnableWindow(m_bInterleave);
   GetDlgItem(IDC_INTERLEAVER_SEED)->EnableWindow(m_bInterleave);
   UpdateDisplay();
   }

void CFilterExtractDlg::OnPresetStrength()
   {
   m_bPresetStrength = ((CButton*)GetDlgItem(IDC_PRESET_STRENGTH))->GetCheck();
   GetDlgItem(IDC_STRENGTH)->EnableWindow(m_bPresetStrength);
   GetDlgItem(IDC_COMPUTE_STRENGTH)->EnableWindow(m_bPresetStrength);
   }

// button events

void CFilterExtractDlg::OnComputeStrength()
   {
   CString sTemp;
   CComputeStrengthDlg dlg;
   // get current strength setting
   GetDlgItemText(IDC_EMBED_STRENGTH, sTemp);
   m_dEmbedStrength = atof(sTemp);
   // compute power equivalent to this strength
   using libbase::round;
   dlg.m_dPower = round(pow(10.0, m_dEmbedStrength / 10.0) * 255*255, 0.1);
   if(dlg.DoModal() == IDOK)
      {
      m_dEmbedStrength = round(10 * log10(dlg.m_dPower / double(255*255)), 0.1);
      sTemp.Format("%g", m_dEmbedStrength);
      SetDlgItemText(IDC_EMBED_STRENGTH, sTemp);
      }
   }

void CFilterExtractDlg::OnLoadSource()
   {
   CFileDialog dlg(TRUE, NULL, "*.*");
   if(dlg.DoModal() == IDOK)
      {
      m_sSource = dlg.GetPathName();
      SetDlgItemText(IDC_SOURCE, m_sSource);
      ComputeFileData();
      UpdateDisplay();
      }
   }

void CFilterExtractDlg::OnLoadCodec()
   {
   CFileDialog dlg(TRUE, NULL, "*.*");
   if(dlg.DoModal() == IDOK)
      {
      m_sCodec = dlg.GetPathName();
      SetDlgItemText(IDC_CODEC, m_sCodec);
      ComputeCodecData();
      UpdateDisplay();
      }
   }

void CFilterExtractDlg::OnLoadPuncture()
   {
   CFileDialog dlg(TRUE, NULL, "*.*");
   if(dlg.DoModal() == IDOK)
      {
      m_sPuncture = dlg.GetPathName();
      SetDlgItemText(IDC_PUNCTURE, m_sPuncture);
      ComputePunctureData();
      UpdateDisplay();
      }
   }

void CFilterExtractDlg::OnClearSource()
   {
   m_sSource = "";
   SetDlgItemText(IDC_SOURCE, m_sSource);
   ComputeFileData();
   UpdateDisplay();
   }

void CFilterExtractDlg::OnClearCodec()
   {
   m_sCodec = "";
   SetDlgItemText(IDC_CODEC, m_sCodec);
   ComputeCodecData();
   UpdateDisplay();
   }

void CFilterExtractDlg::OnClearPuncture()
   {
   m_sPuncture = "";
   SetDlgItemText(IDC_PUNCTURE, m_sPuncture);
   ComputePunctureData();
   UpdateDisplay();
   }

void CFilterExtractDlg::OnSaveResults()
   {
   CFileDialog dlg(FALSE, NULL, "results.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sResults = dlg.GetPathName();
      SetDlgItemText(IDC_RESULTS, m_sResults);
      }
   }

void CFilterExtractDlg::OnSaveEmbeddedImage()
   {
   CFileDialog dlg(FALSE, NULL, "embedded-image.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sEmbeddedImage = dlg.GetPathName();
      SetDlgItemText(IDC_EMBEDDED_IMAGE, m_sEmbeddedImage);
      }
   }

void CFilterExtractDlg::OnSaveExtractedImage()
   {
   CFileDialog dlg(FALSE, NULL, "extracted-image.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sExtractedImage = dlg.GetPathName();
      SetDlgItemText(IDC_EXTRACTED_IMAGE, m_sExtractedImage);
      }
   }

void CFilterExtractDlg::OnSaveEmbedded()
   {
   CFileDialog dlg(FALSE, NULL, "embedded-sigspace.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sEmbedded = dlg.GetPathName();
      SetDlgItemText(IDC_EMBEDDED, m_sEmbedded);
      }
   }

void CFilterExtractDlg::OnSaveExtracted()
   {
   CFileDialog dlg(FALSE, NULL, "extracted-sigspace.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sExtracted = dlg.GetPathName();
      SetDlgItemText(IDC_EXTRACTED, m_sExtracted);
      }
   }

void CFilterExtractDlg::OnSaveDecoded()
   {
   CFileDialog dlg(FALSE, NULL, "decoded.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sDecoded = dlg.GetPathName();
      SetDlgItemText(IDC_DECODED, m_sDecoded);
      }
   }

void CFilterExtractDlg::OnSaveUniform()
   {
   CFileDialog dlg(FALSE, NULL, "uniform.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sUniform = dlg.GetPathName();
      SetDlgItemText(IDC_UNIFORM, m_sUniform);
      }
   }

void CFilterExtractDlg::OnClearResults()
   {
   m_sResults = "";
   SetDlgItemText(IDC_RESULTS, m_sResults);
   }

void CFilterExtractDlg::OnClearEmbeddedImage()
   {
   m_sEmbeddedImage = "";
   SetDlgItemText(IDC_EMBEDDED_IMAGE, m_sEmbeddedImage);
   }

void CFilterExtractDlg::OnClearExtractedImage()
   {
   m_sExtractedImage = "";
   SetDlgItemText(IDC_EXTRACTED_IMAGE, m_sExtractedImage);
   }

void CFilterExtractDlg::OnClearEmbedded()
   {
   m_sEmbedded = "";
   SetDlgItemText(IDC_EMBEDDED, m_sEmbedded);
   }

void CFilterExtractDlg::OnClearExtracted()
   {
   m_sExtracted = "";
   SetDlgItemText(IDC_EXTRACTED, m_sExtracted);
   }

void CFilterExtractDlg::OnClearDecoded()
   {
   m_sDecoded = "";
   SetDlgItemText(IDC_DECODED, m_sDecoded);
   }

void CFilterExtractDlg::OnClearUniform()
   {
   m_sUniform = "";
   SetDlgItemText(IDC_UNIFORM, m_sUniform);
   }

// Helper functions

void CFilterExtractDlg::ComputeFileData()
   {
   if(!m_sSource.IsEmpty())
      {
      CFile file(m_sSource, CFile::modeRead);
      m_nFileSize = int(file.GetLength());
      }
   else
      m_nFileSize = 0;
   }

void CFilterExtractDlg::ComputeCodecData()
   {
   if(!m_sCodec.IsEmpty())
      {
      std::ifstream file(m_sCodec);
      libcomm::codec *pCodec;
      file >> pCodec;
      using libbase::round;
      m_nCodecIn = int(round(pCodec->input_bits()));
      m_nCodecOut = int(round(pCodec->output_bits()));
      delete pCodec;
      }
   else
      {
      m_nCodecIn = 0;
      m_nCodecOut = 0;
      }
   }

void CFilterExtractDlg::ComputePunctureData()
   {
   if(!m_sPuncture.IsEmpty())
      {
      std::ifstream file(m_sPuncture);
      libcomm::puncture *pPuncture;
      file >> pPuncture;
      m_nPunctureIn = pPuncture->get_inputs();
      m_nPunctureOut = pPuncture->get_outputs();
      delete pPuncture;
      }
   else
      {
      m_nPunctureIn = 0;
      m_nPunctureOut = 0;
      }
   }

void CFilterExtractDlg::UpdateDisplay()
   {
   CString sTemp;
   // size of input file, in bytes
   sTemp.Format("%d bytes", m_nFileSize);
   SetDlgItemText(IDC_INFO_FILESIZE, (m_nSourceType == 3 && m_nFileSize > 0) ? sTemp : "");
   // capacity after variable-density interleaving
   const int nCapacity = m_bInterleave ? int(floor(m_nRawSize/m_nEmbedRate * m_dInterleaverDensity)) : int(floor(m_nRawSize/double(m_nEmbedRate)));
   sTemp.Format("%d bits", nCapacity);
   SetDlgItemText(IDC_INFO_CAPACITY, sTemp);
   // code size information
   const int nInSize = m_nCodecIn;
   const int nOutSize = (m_nCodecOut == m_nPunctureIn) ? m_nPunctureOut : m_nCodecOut;
   sTemp.Format("(%d,%d)", nOutSize, nInSize);
   SetDlgItemText(IDC_INFO_CODESIZE, (nInSize > 0) ? sTemp : "");
   // number of usable bits
   const int nUsable = (nInSize > 0) ? int(floor(nCapacity / double(nOutSize))) * nInSize : nCapacity;
   sTemp.Format("%d bits", nUsable);
   SetDlgItemText(IDC_INFO_USABLE, sTemp);
   // data rate in bits per pixel
   sTemp.Format("%g bpp", nUsable/double(m_nRawSize));
   SetDlgItemText(IDC_INFO_DATARATE, sTemp);
   // file usage in percent
   sTemp.Format("%3.1f%%", 100*(m_nFileSize*8)/double(nUsable));
   SetDlgItemText(IDC_INFO_USAGE, (m_nSourceType == 3 && m_nFileSize > 0) ? sTemp : "");
   }
