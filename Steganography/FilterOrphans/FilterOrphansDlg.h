#ifndef afx_filterorphansdlg_h
#define afx_filterorphansdlg_h

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
   CAboutDlg();

   // Dialog Data
   //{{AFX_DATA(CAboutDlg)
   enum { IDD = IDD_ABOUTBOX };
   //}}AFX_DATA

   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAboutDlg)
protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

   // Implementation
protected:
   //{{AFX_MSG(CAboutDlg)
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansDlg dialog

class CFilterOrphansDlg : public CDialog
{
// Construction
public:
   CFilterOrphansDlg(CWnd* pParent = NULL);   // standard constructor

   libwin::CPSPlugIn*  m_pPSPlugIn;

// Dialog Data
   //{{AFX_DATA(CFilterOrphansDlg)
        enum { IDD = IDD_DIALOG1 };
        CSliderCtrl     m_scSlider;
        BOOL    m_bKeepNoise;
        int             m_nWeight;
        //}}AFX_DATA

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterOrphansDlg)
   protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

// Implementation
protected:

   // Generated message map functions
   //{{AFX_MSG(CFilterOrphansDlg)
   virtual BOOL OnInitDialog();
        afx_msg void OnCustomdrawSlider(NMHDR* pNMHDR, LRESULT* pResult);
        afx_msg void OnChangeWeight();
        //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
