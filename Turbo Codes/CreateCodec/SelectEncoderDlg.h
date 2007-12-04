#if !defined(AFX_SELECTENCODERDLG_H__FCB2545A_A78D_411F_A580_AA3713142ED5__INCLUDED_)
#define AFX_SELECTENCODERDLG_H__FCB2545A_A78D_411F_A580_AA3713142ED5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectEncoderDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectEncoderDlg dialog

class CSelectEncoderDlg : public CDialog
{
// Construction
public:
        CSelectEncoderDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectEncoderDlg)
        enum { IDD = IDD_ENCODER };
        int             m_nType;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectEncoderDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectEncoderDlg)
                // NOTE: the ClassWizard will add member functions here
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTENCODERDLG_H__FCB2545A_A78D_411F_A580_AA3713142ED5__INCLUDED_)
