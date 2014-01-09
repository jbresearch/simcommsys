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

#include "FolderDialog.h"

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
// CAutomateEmbeddingDlg dialog

CAutomateEmbeddingDlg::CAutomateEmbeddingDlg(CWnd* pParent /*=NULL*/)
: CDialog(CAutomateEmbeddingDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CAutomateEmbeddingDlg)
        m_nJpegMin = 0;
        m_nJpegMax = 0;
        m_dStrengthMax = 0.0;
        m_dStrengthMin = 0.0;
        m_bJpeg = FALSE;
        m_nJpegStep = 0;
        m_dStrengthStep = 0.0;
        m_sOutput = _T("");
        //}}AFX_DATA_INIT
   }


void CAutomateEmbeddingDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CAutomateEmbeddingDlg)
        DDX_Text(pDX, IDC_JPEGQ_MIN, m_nJpegMin);
        DDX_Text(pDX, IDC_JPEGQ_MAX, m_nJpegMax);
        DDX_Text(pDX, IDC_STRENGTH_MAX, m_dStrengthMax);
        DDX_Text(pDX, IDC_STRENGTH_MIN, m_dStrengthMin);
        DDX_Check(pDX, IDC_JPEG, m_bJpeg);
        DDX_Text(pDX, IDC_JPEGQ_STEP, m_nJpegStep);
        DDX_Text(pDX, IDC_STRENGTH_STEP, m_dStrengthStep);
        DDX_Text(pDX, IDC_OUTPUT, m_sOutput);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CAutomateEmbeddingDlg, CDialog)
//{{AFX_MSG_MAP(CAutomateEmbeddingDlg)
        ON_BN_CLICKED(IDC_JPEG, OnJpeg)
        ON_BN_CLICKED(IDC_OUTPUT_BROWSE, OnOutputBrowse)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingDlg message handlers

BOOL CAutomateEmbeddingDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // make sure all buttons are set up correctly
   GetDlgItem(IDC_JPEGQ_MIN)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_MAX)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_STEP)->EnableWindow(m_bJpeg);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CAutomateEmbeddingDlg::OnOutputBrowse()
   {
   libwin::CFolderDialog dlg("Select output folder:", m_sOutput);
   if(dlg.DoModal() == IDOK)
      {
      m_sOutput = dlg.GetFolder();
      SetDlgItemText(IDC_OUTPUT, m_sOutput);
      }
   }

void CAutomateEmbeddingDlg::OnJpeg()
   {
   m_bJpeg = ((CButton*)GetDlgItem(IDC_JPEG))->GetCheck();
   GetDlgItem(IDC_JPEGQ_MIN)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_MAX)->EnableWindow(m_bJpeg);
   GetDlgItem(IDC_JPEGQ_STEP)->EnableWindow(m_bJpeg);
   }

void CAutomateEmbeddingDlg::OnOK()
   {
   // base class routine
   CDialog::OnOK();
   }
