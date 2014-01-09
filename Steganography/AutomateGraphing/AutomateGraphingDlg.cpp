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
// CAutomateGraphingDlg dialog

CAutomateGraphingDlg::CAutomateGraphingDlg(CWnd* pParent /*=NULL*/)
: CDialog(CAutomateGraphingDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CAutomateGraphingDlg)
        m_nJpegMin = 0;
        m_nJpegMax = 0;
        m_dStrengthMax = 0.0;
        m_dStrengthMin = 0.0;
        m_bJpeg = FALSE;
        m_nJpegStep = 0;
        m_dStrengthStep = 0.0;
        m_sParameters = _T("");
        m_sResults = _T("");
        m_bPresetStrength = FALSE;
        m_bPrintBER = FALSE;
        m_bPrintChiSquare = FALSE;
        m_bPrintEstimate = FALSE;
        m_bPrintSNR = FALSE;
        //}}AFX_DATA_INIT
   }


void CAutomateGraphingDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CAutomateGraphingDlg)
        DDX_Text(pDX, IDC_JPEGQ_MIN, m_nJpegMin);
        DDX_Text(pDX, IDC_JPEGQ_MAX, m_nJpegMax);
        DDX_Text(pDX, IDC_STRENGTH_MAX, m_dStrengthMax);
        DDX_Text(pDX, IDC_STRENGTH_MIN, m_dStrengthMin);
        DDX_Check(pDX, IDC_JPEG, m_bJpeg);
        DDX_Text(pDX, IDC_JPEGQ_STEP, m_nJpegStep);
        DDX_Text(pDX, IDC_STRENGTH_STEP, m_dStrengthStep);
        DDX_Text(pDX, IDC_PARAMETERS, m_sParameters);
        DDX_Text(pDX, IDC_RESULTS, m_sResults);
        DDX_Check(pDX, IDC_PRESET_STRENGTH, m_bPresetStrength);
        DDX_Check(pDX, IDC_PRINT_BER, m_bPrintBER);
        DDX_Check(pDX, IDC_PRINT_CHI_SQUARE, m_bPrintChiSquare);
        DDX_Check(pDX, IDC_PRINT_ESTIMATE, m_bPrintEstimate);
        DDX_Check(pDX, IDC_PRINT_SNR, m_bPrintSNR);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CAutomateGraphingDlg, CDialog)
//{{AFX_MSG_MAP(CAutomateGraphingDlg)
        ON_BN_CLICKED(IDC_PARAMETERS_BROWSE, OnParametersBrowse)
        ON_BN_CLICKED(IDC_RESULTS_BROWSE, OnResultsBrowse)
        ON_BN_CLICKED(IDC_JPEG, OnJpeg)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingDlg message handlers

BOOL CAutomateGraphingDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // make sure all buttons are set up correctly
   GetDlgItem(IDC_JPEGQ_MIN)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_MAX)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_STEP)->EnableWindow(m_bJpeg);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CAutomateGraphingDlg::OnParametersBrowse()
   {
   CFileDialog dlg(TRUE, NULL, m_sParameters);
   if(dlg.DoModal() == IDOK)
      {
      m_sParameters = dlg.GetPathName();
      SetDlgItemText(IDC_PARAMETERS, m_sParameters);
      }
   }

void CAutomateGraphingDlg::OnResultsBrowse()
   {
   CFileDialog dlg(FALSE, NULL, m_sResults.IsEmpty() ? "results.txt" : m_sResults);
   if(dlg.DoModal() == IDOK)
      {
      m_sResults = dlg.GetPathName();
      SetDlgItemText(IDC_RESULTS, m_sResults);
      }
   }

void CAutomateGraphingDlg::OnJpeg()
   {
   m_bJpeg = ((CButton*)GetDlgItem(IDC_JPEG))->GetCheck();
   GetDlgItem(IDC_JPEGQ_MIN)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_MAX)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_STEP)->EnableWindow(m_bJpeg);
   }

void CAutomateGraphingDlg::OnOK()
   {
   // extra validation
   if(!m_bPrintBER && !m_bPrintSNR && !m_bPrintEstimate && !m_bPrintChiSquare)
      {
      MessageBox("You must select something to output!");
      return;
      }
   // base class routine
   CDialog::OnOK();
   }
