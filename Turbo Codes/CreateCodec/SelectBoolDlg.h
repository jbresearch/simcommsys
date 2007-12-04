#if !defined(AFX_SELECTBOOLDLG_H__A70157DD_B1B1_483D_A57F_357760A06937__INCLUDED_)
#define AFX_SELECTBOOLDLG_H__A70157DD_B1B1_483D_A57F_357760A06937__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectBoolDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectBoolDlg dialog

class CSelectBoolDlg : public CDialog
{
// Construction
public:
        CSelectBoolDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectBoolDlg)
        enum { IDD = IDD_BOOL };
        int             m_nValue;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectBoolDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectBoolDlg)
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTBOOLDLG_H__A70157DD_B1B1_483D_A57F_357760A06937__INCLUDED_)
