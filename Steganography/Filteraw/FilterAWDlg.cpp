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
#include "FilterAW.h"
#include "FilterAWDlg.h"

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
// CFilterAWDlg dialog

CFilterAWDlg::CFilterAWDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterAWDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterAWDlg)
        m_nRadius = 0;
        m_bKeepNoise = FALSE;
        m_dNoise = 0.0;
        m_bAuto = FALSE;
        //}}AFX_DATA_INIT
   }


void CFilterAWDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterAWDlg)
        DDX_Text(pDX, IDC_RADIUS, m_nRadius);
        DDX_Check(pDX, IDC_KEEPNOISE, m_bKeepNoise);
        DDX_Text(pDX, IDC_NOISE, m_dNoise);
        DDX_Check(pDX, IDC_AUTO, m_bAuto);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterAWDlg, CDialog)
//{{AFX_MSG_MAP(CFilterAWDlg)
        ON_BN_CLICKED(IDC_AUTO, OnAuto)
        ON_EN_CHANGE(IDC_RADIUS, OnChangeRadius)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterAWDlg message handlers

BOOL CFilterAWDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // set region (cf OnChangeRadius)
   CString sTemp;
   sTemp.Format(" Region: %dx%d", 2*m_nRadius+1, 2*m_nRadius+1);
   SetDlgItemText(IDC_REGION, sTemp);
   // set noise (cf OnAuto)
   GetDlgItem(IDC_NOISE)->EnableWindow(!m_bAuto);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterAWDlg::OnChangeRadius()
   {
   m_nRadius = GetDlgItemInt(IDC_RADIUS);
   CString sTemp;
   sTemp.Format(" Region: %dx%d", 2*m_nRadius+1, 2*m_nRadius+1);
   SetDlgItemText(IDC_REGION, sTemp);
   }

void CFilterAWDlg::OnAuto()
   {
   m_bAuto = ((CButton*)GetDlgItem(IDC_AUTO))->GetCheck();
   GetDlgItem(IDC_NOISE)->EnableWindow(!m_bAuto);
   }
