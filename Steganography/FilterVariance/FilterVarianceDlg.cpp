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
#include "FilterVariance.h"
#include "FilterVarianceDlg.h"

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
// CFilterVarianceDlg dialog

CFilterVarianceDlg::CFilterVarianceDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterVarianceDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterVarianceDlg)
        m_nRadius = 0;
        m_nScale = 0;
        m_bAutoScale = FALSE;
        //}}AFX_DATA_INIT
   }


void CFilterVarianceDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterVarianceDlg)
        DDX_Text(pDX, IDC_RADIUS, m_nRadius);
        DDX_Text(pDX, IDC_SCALE, m_nScale);
        DDX_Check(pDX, IDC_AUTOSCALE, m_bAutoScale);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterVarianceDlg, CDialog)
//{{AFX_MSG_MAP(CFilterVarianceDlg)
        ON_EN_CHANGE(IDC_RADIUS, OnChangeRadius)
        ON_BN_CLICKED(IDC_AUTOSCALE, OnAutoscale)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterVarianceDlg message handlers

BOOL CFilterVarianceDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // set region (cf OnChangeRadius)
   CString sTemp;
   sTemp.Format(" Region: %dx%d", 2*m_nRadius+1, 2*m_nRadius+1);
   SetDlgItemText(IDC_REGION, sTemp);
   // set scale (cf OnAutoscale)
   GetDlgItem(IDC_SCALE)->EnableWindow(!m_bAutoScale);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterVarianceDlg::OnChangeRadius()
   {
   m_nRadius = GetDlgItemInt(IDC_RADIUS);
   CString sTemp;
   sTemp.Format(" Region: %dx%d", 2*m_nRadius+1, 2*m_nRadius+1);
   SetDlgItemText(IDC_REGION, sTemp);
   }

void CFilterVarianceDlg::OnAutoscale()
   {
   m_bAutoScale = ((CButton*)GetDlgItem(IDC_AUTOSCALE))->GetCheck();
   GetDlgItem(IDC_SCALE)->EnableWindow(!m_bAutoScale);
   }
