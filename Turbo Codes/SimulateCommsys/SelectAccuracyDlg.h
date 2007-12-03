#if !defined(AFX_SELECTACCURACYDLG_H__B4CB187D_B147_4F82_AA21_609AF1F3116A__INCLUDED_)
#define AFX_SELECTACCURACYDLG_H__B4CB187D_B147_4F82_AA21_609AF1F3116A__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectAccuracyDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectAccuracyDlg dialog

class CSelectAccuracyDlg : public CDialog
{
// Construction
public:
        CSelectAccuracyDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectAccuracyDlg)
        enum { IDD = IDD_ACCURACY };
        double  m_dAccuracy;
        double  m_dConfidence;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectAccuracyDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectAccuracyDlg)
        virtual void OnOK();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTACCURACYDLG_H__B4CB187D_B147_4F82_AA21_609AF1F3116A__INCLUDED_)
