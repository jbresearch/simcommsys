#if !defined(AFX_SELECTRANGEDLG_H__9DEFE8D5_6EAE_4D81_988A_5A25A52F14B9__INCLUDED_)
#define AFX_SELECTRANGEDLG_H__9DEFE8D5_6EAE_4D81_988A_5A25A52F14B9__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectRangeDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectRangeDlg dialog

class CSelectRangeDlg : public CDialog
{
// Construction
public:
        CSelectRangeDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectRangeDlg)
        enum { IDD = IDD_RANGE };
        double  m_dSNRmax;
        double  m_dSNRmin;
        double  m_dSNRstep;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectRangeDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectRangeDlg)
        virtual void OnOK();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTRANGEDLG_H__9DEFE8D5_6EAE_4D81_988A_5A25A52F14B9__INCLUDED_)
