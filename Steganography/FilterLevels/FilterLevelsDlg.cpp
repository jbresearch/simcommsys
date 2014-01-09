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
#include "FilterLevels.h"
#include "FilterLevelsDlg.h"

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
// CFilterLevelsDlg dialog

CFilterLevelsDlg::CFilterLevelsDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterLevelsDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterLevelsDlg)
        m_bKeepNoise = FALSE;
        m_nWeight = 0;
        //}}AFX_DATA_INIT
   }


void CFilterLevelsDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterLevelsDlg)
        DDX_Control(pDX, IDC_SLIDER, m_scSlider);
        DDX_Check(pDX, IDC_KEEPNOISE, m_bKeepNoise);
        DDX_Text(pDX, IDC_WEIGHT, m_nWeight);
        DDV_MinMaxInt(pDX, m_nWeight, 0, 8);
        //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterLevelsDlg, CDialog)
//{{AFX_MSG_MAP(CFilterLevelsDlg)
        ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER, OnCustomdrawSlider)
        ON_EN_CHANGE(IDC_WEIGHT, OnChangeWeight)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsDlg message handlers

BOOL CFilterLevelsDlg::OnInitDialog()
   {
   CDialog::OnInitDialog();

   // set slider limits & position (cf OnChangeWeight)
   m_scSlider.SetRange(0, 8);
   m_scSlider.SetPos(m_nWeight);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterLevelsDlg::OnChangeWeight()
   {
   m_nWeight = GetDlgItemInt(IDC_WEIGHT);
   if(m_scSlider.GetPos() != m_nWeight)
      {
      m_scSlider.SetPos(m_nWeight);
      m_scSlider.RedrawWindow();
      }
   }

void CFilterLevelsDlg::OnCustomdrawSlider(NMHDR* pNMHDR, LRESULT* pResult)
   {
   if(m_scSlider.GetPos() != m_nWeight && m_scSlider.GetRangeMax() >= m_nWeight)
      SetDlgItemInt(IDC_WEIGHT, m_scSlider.GetPos());
   *pResult = 0;
   }
