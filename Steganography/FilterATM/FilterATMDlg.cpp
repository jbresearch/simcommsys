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
#include "FilterATM.h"
#include "FilterATMDlg.h"

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
// CFilterATMDlg dialog

CFilterATMDlg::CFilterATMDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterATMDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterATMDlg)
        m_nAlpha = 0;
        m_nRadius = 0;
        m_bKeepNoise = FALSE;
        //}}AFX_DATA_INIT
   }


void CFilterATMDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterATMDlg)
        DDX_Control(pDX, IDC_SLIDER, m_scSlider);
        DDX_Text(pDX, IDC_ALPHA, m_nAlpha);
        DDX_Text(pDX, IDC_RADIUS, m_nRadius);
        DDX_Check(pDX, IDC_KEEPNOISE, m_bKeepNoise);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterATMDlg, CDialog)
//{{AFX_MSG_MAP(CFilterATMDlg)
        ON_EN_CHANGE(IDC_RADIUS, OnChangeRadius)
        ON_EN_CHANGE(IDC_ALPHA, OnChangeAlpha)
        ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER, OnCustomdrawSlider)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterATMDlg message handlers

BOOL CFilterATMDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // set region (cf OnChangeRadius)
   CString sTemp;
   sTemp.Format(" Region: %dx%d", 2*m_nRadius+1, 2*m_nRadius+1);
   SetDlgItemText(IDC_REGION, sTemp);
   m_scSlider.SetRange(0, 2*m_nRadius*(m_nRadius+1));
   m_scSlider.SetPos(m_nAlpha);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterATMDlg::OnChangeRadius()
   {
   m_nRadius = GetDlgItemInt(IDC_RADIUS);
   CString sTemp;
   sTemp.Format(" Region: %dx%d", 2*m_nRadius+1, 2*m_nRadius+1);
   SetDlgItemText(IDC_REGION, sTemp);
   m_scSlider.SetRange(0, 2*m_nRadius*(m_nRadius+1));
   m_scSlider.SetPos(m_nAlpha);
   m_scSlider.RedrawWindow();
   }

void CFilterATMDlg::OnChangeAlpha()
   {
   m_nAlpha = GetDlgItemInt(IDC_ALPHA);
   if(m_scSlider.GetPos() != m_nAlpha)
      {
      m_scSlider.SetPos(m_nAlpha);
      m_scSlider.RedrawWindow();
      }
   }

void CFilterATMDlg::OnCustomdrawSlider(NMHDR* pNMHDR, LRESULT* pResult)
   {
   if(m_scSlider.GetPos() != m_nAlpha && m_scSlider.GetRangeMax() >= m_nAlpha)
      SetDlgItemInt(IDC_ALPHA, m_scSlider.GetPos());
   *pResult = 0;
   }
