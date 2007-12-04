#if !defined(AFX_SELECTMODULATORDLG_H__018DFBA3_31DA_4671_8D14_B532F7052D62__INCLUDED_)
#define AFX_SELECTMODULATORDLG_H__018DFBA3_31DA_4671_8D14_B532F7052D62__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectModulatorDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectModulatorDlg dialog

class CSelectModulatorDlg : public CDialog
{
// Construction
public:
        CSelectModulatorDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectModulatorDlg)
        enum { IDD = IDD_MODULATOR };
        int             m_nType;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectModulatorDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectModulatorDlg)
                // NOTE: the ClassWizard will add member functions here
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTMODULATORDLG_H__018DFBA3_31DA_4671_8D14_B532F7052D62__INCLUDED_)
