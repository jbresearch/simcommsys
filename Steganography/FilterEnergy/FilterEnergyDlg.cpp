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
// CFilterEnergyDlg dialog

CFilterEnergyDlg::CFilterEnergyDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterEnergyDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterEnergyDlg)
        m_bAppend = FALSE;
        m_bDisplayEnergy = FALSE;
        m_bDisplayPixelCount = FALSE;
        m_bDisplayVariance = FALSE;
        m_sFileName = _T("");
        m_bScreenOnly = FALSE;
        //}}AFX_DATA_INIT
   }


void CFilterEnergyDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterEnergyDlg)
        DDX_Check(pDX, IDC_APPEND, m_bAppend);
        DDX_Check(pDX, IDC_DISPLAY_ENERGY, m_bDisplayEnergy);
        DDX_Check(pDX, IDC_DISPLAY_PIXELCOUNT, m_bDisplayPixelCount);
        DDX_Check(pDX, IDC_DISPLAY_VARIANCE, m_bDisplayVariance);
        DDX_Text(pDX, IDC_FILENAME, m_sFileName);
        DDX_Check(pDX, IDC_SCREENONLY, m_bScreenOnly);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterEnergyDlg, CDialog)
//{{AFX_MSG_MAP(CFilterEnergyDlg)
        ON_BN_CLICKED(IDC_SCREENONLY, OnScreenOnly)
        ON_BN_CLICKED(IDC_BROWSE, OnBrowse)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterEnergyDlg message handlers

BOOL CFilterEnergyDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // set scale (cf OnScreenOnly)
   GetDlgItem(IDC_FILENAME)->EnableWindow(!m_bScreenOnly);
   GetDlgItem(IDC_BROWSE)->EnableWindow(!m_bScreenOnly);
   GetDlgItem(IDC_APPEND)->EnableWindow(!m_bScreenOnly);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterEnergyDlg::OnScreenOnly()
   {
   m_bScreenOnly = ((CButton*)GetDlgItem(IDC_SCREENONLY))->GetCheck();
   GetDlgItem(IDC_FILENAME)->EnableWindow(!m_bScreenOnly);
   GetDlgItem(IDC_BROWSE)->EnableWindow(!m_bScreenOnly);
   GetDlgItem(IDC_APPEND)->EnableWindow(!m_bScreenOnly);
   }

void CFilterEnergyDlg::OnBrowse()
   {
   CFileDialog dlg(FALSE, NULL, m_sFileName.IsEmpty() ? "results.txt" : m_sFileName);
   if(dlg.DoModal() == IDOK)
      {
      m_sFileName = dlg.GetPathName();
      SetDlgItemText(IDC_FILENAME, m_sFileName);
      }
   }
