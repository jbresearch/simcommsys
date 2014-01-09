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

// AnalyseInterleaverDlg.h : header file
//

#if !defined(AFX_ANALYSEINTERLEAVERDLG_H__797804D0_A1D8_4400_B11C_1A298E94C19E__INCLUDED_)
#define AFX_ANALYSEINTERLEAVERDLG_H__797804D0_A1D8_4400_B11C_1A298E94C19E__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "interleaver.h"
#include "interleaver/lut/named/file_lut.h"
#include "interleaver/lut/named/stream_lut.h"
#include "matrix.h"
#include "vector.h"

/////////////////////////////////////////////////////////////////////////////
// CAnalyseInterleaverDlg dialog

class CAnalyseInterleaverDlg : public CDialog
{
// Construction
public:
        CAnalyseInterleaverDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CAnalyseInterleaverDlg)
        enum { IDD = IDD_ANALYSEINTERLEAVER_DIALOG };
        CProgressCtrl   m_pcProgress;
        CString m_sPathName;
        int             m_nTau;
        int             m_nSpread;
        int             m_nMaxDist;
        //}}AFX_DATA

        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CAnalyseInterleaverDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);        // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:
   libcomm::interleaver<double>* m_pInterleaver;
   libbase::matrix<int> m_miIOSS;
        HICON m_hIcon;

        // Generated message map functions
        //{{AFX_MSG(CAnalyseInterleaverDlg)
        virtual BOOL OnInitDialog();
        afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
        afx_msg void OnPaint();
        afx_msg HCURSOR OnQueryDragIcon();
        afx_msg void OnLoad();
        afx_msg void OnAnalyse();
        virtual void OnOK();
        virtual void OnCancel();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ANALYSEINTERLEAVERDLG_H__797804D0_A1D8_4400_B11C_1A298E94C19E__INCLUDED_)
