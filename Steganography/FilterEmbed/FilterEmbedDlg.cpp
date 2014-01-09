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
#include "FilterEmbed.h"
#include "FilterEmbedDlg.h"
#include "ComputeStrengthDlg.h"

#include <fstream>
using namespace std;

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
// CFilterEmbedDlg dialog

CFilterEmbedDlg::CFilterEmbedDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterEmbedDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterEmbedDlg)
        m_sCodec = _T("");
        m_sPuncture = _T("");
        m_sSource = _T("");
        m_nSourceSeed = 0;
        m_nSourceType = -1;
        m_bInterleave = FALSE;
        m_nInterleaverSeed = 0;
        m_dInterleaverDensity = 0.0;
        m_nEmbedSeed = 0;
        m_dEmbedStrength = 0.0;
        m_nEmbedRate = 0;
        //}}AFX_DATA_INIT
   }


void CFilterEmbedDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterEmbedDlg)
        DDX_Text(pDX, IDC_CODEC, m_sCodec);
        DDX_Text(pDX, IDC_PUNCTURE, m_sPuncture);
        DDX_Text(pDX, IDC_SOURCE, m_sSource);
        DDX_Text(pDX, IDC_SOURCE_SEED, m_nSourceSeed);
        DDX_CBIndex(pDX, IDC_SOURCE_TYPE, m_nSourceType);
        DDX_Check(pDX, IDC_INTERLEAVE, m_bInterleave);
        DDX_Text(pDX, IDC_INTERLEAVER_SEED, m_nInterleaverSeed);
        DDX_Text(pDX, IDC_INTERLEAVER_DENSITY, m_dInterleaverDensity);
        DDX_Text(pDX, IDC_EMBED_SEED, m_nEmbedSeed);
        DDX_Text(pDX, IDC_EMBED_STRENGTH, m_dEmbedStrength);
        DDX_Text(pDX, IDC_EMBED_RATE, m_nEmbedRate);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterEmbedDlg, CDialog)
//{{AFX_MSG_MAP(CFilterEmbedDlg)
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
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterEmbedDlg message handlers

BOOL CFilterEmbedDlg::OnInitDialog()
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

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterEmbedDlg::OnOK()
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

void CFilterEmbedDlg::OnChangeEmbedRate()
   {
   m_nEmbedRate = GetDlgItemInt(IDC_EMBED_RATE);
   if(m_nEmbedRate < 1)
      m_nEmbedRate = 1;
   UpdateDisplay();
   }

void CFilterEmbedDlg::OnChangeInterleaverDensity()
   {
   CString sTemp;
   GetDlgItemText(IDC_INTERLEAVER_DENSITY, sTemp);
   m_dInterleaverDensity = atof(sTemp);
   UpdateDisplay();
   }

void CFilterEmbedDlg::OnSelchangeSourceType()
   {
   UpdateData(true);
   //m_nSourceType = GetDlgItemInt(IDC_SOURCE_TYPE);
   GetDlgItem(IDC_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_CLEAR_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_LOAD_SOURCE)->EnableWindow(m_nSourceType==3);
   GetDlgItem(IDC_SOURCE_SEED)->EnableWindow(m_nSourceType==2);
   UpdateDisplay();
   }

void CFilterEmbedDlg::OnInterleave()
   {
   m_bInterleave = ((CButton*)GetDlgItem(IDC_INTERLEAVE))->GetCheck();
   GetDlgItem(IDC_INTERLEAVER_DENSITY)->EnableWindow(m_bInterleave);
   GetDlgItem(IDC_INTERLEAVER_SEED)->EnableWindow(m_bInterleave);
   UpdateDisplay();
   }

// button events

void CFilterEmbedDlg::OnComputeStrength()
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

void CFilterEmbedDlg::OnLoadSource()
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

void CFilterEmbedDlg::OnLoadCodec()
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

void CFilterEmbedDlg::OnLoadPuncture()
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

void CFilterEmbedDlg::OnClearSource()
   {
   m_sSource = "";
   SetDlgItemText(IDC_SOURCE, m_sSource);
   ComputeFileData();
   UpdateDisplay();
   }

void CFilterEmbedDlg::OnClearCodec()
   {
   m_sCodec = "";
   SetDlgItemText(IDC_CODEC, m_sCodec);
   ComputeCodecData();
   UpdateDisplay();
   }

void CFilterEmbedDlg::OnClearPuncture()
   {
   m_sPuncture = "";
   SetDlgItemText(IDC_PUNCTURE, m_sPuncture);
   ComputePunctureData();
   UpdateDisplay();
   }

// Helper functions

void CFilterEmbedDlg::ComputeFileData()
   {
   if(!m_sSource.IsEmpty())
      {
      CFile file(m_sSource, CFile::modeRead);
      m_nFileSize = int(file.GetLength());
      }
   else
      m_nFileSize = 0;
   }

void CFilterEmbedDlg::ComputeCodecData()
   {
   if(!m_sCodec.IsEmpty())
      {
      ifstream file(m_sCodec);
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

void CFilterEmbedDlg::ComputePunctureData()
   {
   if(!m_sPuncture.IsEmpty())
      {
      ifstream file(m_sPuncture);
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

void CFilterEmbedDlg::UpdateDisplay()
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
