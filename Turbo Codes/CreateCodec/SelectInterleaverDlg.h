#if !defined(AFX_SELECTINTERLEAVERDLG_H__8AD71EA4_B002_46C3_9D18_841545D612E0__INCLUDED_)
#define AFX_SELECTINTERLEAVERDLG_H__8AD71EA4_B002_46C3_9D18_841545D612E0__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectInterleaverDlg.h : header file
//

#include "interleaver.h"

/////////////////////////////////////////////////////////////////////////////
// CSelectInterleaverDlg dialog

class CSelectInterleaverDlg : public CDialog
{
// Construction
public:
   libcomm::interleaver* m_pInterleaver;
        CSelectInterleaverDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectInterleaverDlg)
        enum { IDD = IDD_INTERLEAVER };
        int             m_nType;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectInterleaverDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectInterleaverDlg)
        afx_msg void OnSelchangeType();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTINTERLEAVERDLG_H__8AD71EA4_B002_46C3_9D18_841545D612E0__INCLUDED_)
