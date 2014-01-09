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
#include "FilterWavelet.h"
#include "FilterWaveletDlg.h"
#include "itfunc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

// set up library names
using std::cerr;
using libbase::trace;

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
// CFilterWaveletDlg dialog

CFilterWaveletDlg::CFilterWaveletDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterWaveletDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterWaveletDlg)
        m_bKeepNoise = FALSE;
        m_nTileX = 0;
        m_nTileY = 0;
        m_bWholeImage = FALSE;
        m_dThreshCutoff = 0.0;
        m_nThreshSelector = -1;
        m_nThreshType = -1;
        m_nWaveletType = -1;
        m_nWaveletLevel = 0;
        m_nWaveletPar = -1;
        //}}AFX_DATA_INIT
   }


void CFilterWaveletDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterWaveletDlg)
        DDX_Check(pDX, IDC_KEEPNOISE, m_bKeepNoise);
        DDX_Text(pDX, IDC_TILEX, m_nTileX);
        DDX_Text(pDX, IDC_TILEY, m_nTileY);
        DDX_Check(pDX, IDC_WHOLEIMAGE, m_bWholeImage);
        DDX_Text(pDX, IDC_THRESH_CUTOFF, m_dThreshCutoff);
        DDX_CBIndex(pDX, IDC_THRESH_SELECTOR, m_nThreshSelector);
        DDX_CBIndex(pDX, IDC_THRESH_TYPE, m_nThreshType);
        DDX_CBIndex(pDX, IDC_WAVELET_TYPE, m_nWaveletType);
        DDX_Text(pDX, IDC_WAVELET_LEVEL, m_nWaveletLevel);
        DDX_CBIndex(pDX, IDC_WAVELET_PAR, m_nWaveletPar);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterWaveletDlg, CDialog)
//{{AFX_MSG_MAP(CFilterWaveletDlg)
        ON_BN_CLICKED(IDC_WHOLEIMAGE, OnWholeimage)
        ON_CBN_SELCHANGE(IDC_THRESH_SELECTOR, OnSelchangeThreshSelector)
        ON_CBN_SELCHANGE(IDC_WAVELET_TYPE, OnSelchangeWaveletType)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletDlg message handlers

BOOL CFilterWaveletDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // set whole-image criterion (cf OnWholeimage)
   GetDlgItem(IDC_TILEX)->EnableWindow(!m_bWholeImage);
   GetDlgItem(IDC_TILEY)->EnableWindow(!m_bWholeImage);
   // set threshold selector criterion (cf OnSelchangeThreshSelector)
   GetDlgItem(IDC_THRESH_CUTOFF)->EnableWindow(m_nThreshSelector==0);
   // set wavelet type criterion (cf OnSelchangeWaveletType)
   SetupWaveletPar();

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterWaveletDlg::OnOK()
   {
   UpdateData(true);
   // set up library names
   using libbase::weight;
   using libbase::log2;
   // handle errors first
   const int nMaxLevel = int(floor(log2(m_bWholeImage ? min(m_pPSPlugIn->GetImageWidth(), m_pPSPlugIn->GetImageHeight()) : min(m_nTileX, m_nTileY)))) - 1;
   if(m_nWaveletLevel < 1 || m_nWaveletLevel > nMaxLevel)
      {
      cerr << "Invalid wavelet level (" << m_nWaveletLevel << "): must be between 1 and " << nMaxLevel << ".\n";
      return;
      }
   if(!m_bWholeImage && (m_nTileX < 8 || m_nTileY < 8))
      {
      cerr << "Invalid tile size (" << m_nTileX << "x" << m_nTileY << "): must be at least 8x8.\n";
      return;
      }
   switch(m_nThreshSelector)
      {
      case 1: // Visu
      case 0: // % of coefficients
         break;
      case 4: // Hybrid+
      case 3: // SURE
      case 2: // Minimax
      default: // unsupported type
         cerr << "Unknown threshold selector (" << m_nThreshSelector << ").\n";
         return;
      }
   // handle warnings next
   if(!m_bWholeImage && (weight(m_nTileX)!=1 || weight(m_nTileY)!=1))
      {
      CString sMessage = "Chosen tile size is not an integral power of two; this could lead to artifacts and slower operation - are you sure you want to use this value?";
      if(MessageBox(sMessage, NULL, MB_YESNO | MB_ICONWARNING) != IDYES)
         return;
      }
   // go ahead
   CDialog::OnOK();
   // convert parameters that need conversion
   m_nWaveletPar = GetDlgItemInt(IDC_WAVELET_PAR);
   trace << "Wavelet parameter converted to: " << m_nWaveletPar << "\n";
   }

void CFilterWaveletDlg::OnWholeimage()
   {
   m_bWholeImage = ((CButton*)GetDlgItem(IDC_WHOLEIMAGE))->GetCheck();
   GetDlgItem(IDC_TILEX)->EnableWindow(!m_bWholeImage);
   GetDlgItem(IDC_TILEY)->EnableWindow(!m_bWholeImage);
   }

void CFilterWaveletDlg::OnSelchangeThreshSelector()
   {
   UpdateData(true);
   GetDlgItem(IDC_THRESH_CUTOFF)->EnableWindow(m_nThreshSelector==0);
   }

void CFilterWaveletDlg::OnSelchangeWaveletType()
   {
   UpdateData(true);
   m_nWaveletPar = GetDlgItemInt(IDC_WAVELET_PAR);
   SetupWaveletPar();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterWaveletDlg helper functions

void CFilterWaveletDlg::SetupWaveletPar()
   {
   GetDlgItem(IDC_WAVELET_PAR)->EnableWindow(m_nWaveletType>=2 && m_nWaveletType!=5);
   CComboBox* p = (CComboBox*)GetDlgItem(IDC_WAVELET_PAR);
   p->ResetContent();
   CString sTemp;
   switch(m_nWaveletType)
      {
      case 5: // Vaidyanathan
      case 0: // Haar
      case 1: // Beylkin
         p->AddString("...");
         break;
      case 2: { // Coiflet
         for(int i=1; i<=5; i++)
            {
            sTemp.Format("%d", i);
            p->AddString(sTemp);
            }
         } break;
      case 3: { // Daubechies
         for(int i=4; i<=20; i+=2)
            {
            sTemp.Format("%d", i);
            p->AddString(sTemp);
            }
         } break;
      case 4: { // Symmlet
         for(int i=4; i<=10; i++)
            {
            sTemp.Format("%d", i);
            p->AddString(sTemp);
            }
         } break;
      case 6: { // Battle-Lemarie
         for(int i=1; i<=5; i+=2)
            {
            sTemp.Format("%d", i);
            p->AddString(sTemp);
            }
         } break;
      default: // Undefined
         cerr << "Undefined wavelet type (" << m_nWaveletType << ").\n";
         return;
      }
   // try to set the new wavelet parameter setting to be the same as the old one
   sTemp.Format("%d", m_nWaveletPar);
   m_nWaveletPar = p->FindStringExact(-1,sTemp);
   if(m_nWaveletPar == CB_ERR)
      m_nWaveletPar = 0;
   p->SetCurSel(m_nWaveletPar);
   }
