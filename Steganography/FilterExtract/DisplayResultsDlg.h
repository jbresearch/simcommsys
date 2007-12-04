#ifndef afx_displayresultsdlg_h
#define afx_displayresultsdlg_h

/////////////////////////////////////////////////////////////////////////////
// CDisplayResultsDlg dialog

class CDisplayResultsDlg : public CDialog
{
// Construction
public:
        CDisplayResultsDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CDisplayResultsDlg)
        enum { IDD = IDD_DISPLAY_RESULTS };
        CString m_sBER;
        CString m_sRate;
        CString m_sSNR;
        CString m_sSNRest;
        CString m_sChiSquare;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CDisplayResultsDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CDisplayResultsDlg)
                // NOTE: the ClassWizard will add member functions here
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
