#ifndef afx_computestrengthdlg_h
#define afx_computestrengthdlg_h

/////////////////////////////////////////////////////////////////////////////
// CComputeStrengthDlg dialog

class CComputeStrengthDlg : public CDialog
{
// Construction
public:
        CComputeStrengthDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CComputeStrengthDlg)
        enum { IDD = IDD_COMPUTE_STRENGTH };
        double  m_dPower;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CComputeStrengthDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CComputeStrengthDlg)
                // NOTE: the ClassWizard will add member functions here
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
