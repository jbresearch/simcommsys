#ifndef afx_filterawdlg_h
#define afx_filterawdlg_h

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
// CFilterAWDlg dialog

class CFilterAWDlg : public CDialog
{
// Construction
public:
   CFilterAWDlg(CWnd* pParent = NULL);   // standard constructor

   libwin::CPSPlugIn*  m_pPSPlugIn;

// Dialog Data
   //{{AFX_DATA(CFilterAWDlg)
        enum { IDD = IDD_DIALOG1 };
        int             m_nRadius;
        BOOL    m_bKeepNoise;
        double  m_dNoise;
        BOOL    m_bAuto;
        //}}AFX_DATA

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterAWDlg)
   protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

// Implementation
protected:

   // Generated message map functions
   //{{AFX_MSG(CFilterAWDlg)
   virtual BOOL OnInitDialog();
        afx_msg void OnAuto();
        afx_msg void OnChangeRadius();
        //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif



