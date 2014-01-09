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
#include "filterextract.h"
#include "ComputeStrengthDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CComputeStrengthDlg dialog


CComputeStrengthDlg::CComputeStrengthDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CComputeStrengthDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CComputeStrengthDlg)
        m_dPower = 0.0;
        //}}AFX_DATA_INIT
}


void CComputeStrengthDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CComputeStrengthDlg)
        DDX_Text(pDX, IDC_POWER, m_dPower);
        DDV_MinMaxDouble(pDX, m_dPower, 0., 65025.);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CComputeStrengthDlg, CDialog)
        //{{AFX_MSG_MAP(CComputeStrengthDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CComputeStrengthDlg message handlers
