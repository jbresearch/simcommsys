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
#include "DisplayResultsDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CDisplayResultsDlg dialog


CDisplayResultsDlg::CDisplayResultsDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CDisplayResultsDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CDisplayResultsDlg)
        m_sBER = _T("");
        m_sRate = _T("");
        m_sSNR = _T("");
        m_sSNRest = _T("");
        m_sChiSquare = _T("");
        //}}AFX_DATA_INIT
}


void CDisplayResultsDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CDisplayResultsDlg)
        DDX_Text(pDX, IDC_BER, m_sBER);
        DDX_Text(pDX, IDC_RATE, m_sRate);
        DDX_Text(pDX, IDC_SNR, m_sSNR);
        DDX_Text(pDX, IDC_SNR_EST, m_sSNRest);
        DDX_Text(pDX, IDC_CHI_SQUARE, m_sChiSquare);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CDisplayResultsDlg, CDialog)
        //{{AFX_MSG_MAP(CDisplayResultsDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CDisplayResultsDlg message handlers
