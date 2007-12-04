#ifndef afx_filterwaveletdlg_h
#define afx_filterwaveletdlg_h

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
// CFilterWaveletDlg dialog

class CFilterWaveletDlg : public CDialog
{
// Construction
public:
   CFilterWaveletDlg(CWnd* pParent = NULL);   // standard constructor

   libwin::CPSPlugIn*  m_pPSPlugIn;

// Dialog Data
   //{{AFX_DATA(CFilterWaveletDlg)
        enum { IDD = IDD_DIALOG1 };
        BOOL    m_bKeepNoise;
        int             m_nTileX;
        int             m_nTileY;
        BOOL    m_bWholeImage;
        double  m_dThreshCutoff;
        int             m_nThreshSelector;
        int             m_nThreshType;
        int             m_nWaveletType;
        int             m_nWaveletLevel;
        int             m_nWaveletPar;
        //}}AFX_DATA

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterWaveletDlg)
   protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

// Implementation
protected:
   void SetupWaveletPar();

   // Generated message map functions
   //{{AFX_MSG(CFilterWaveletDlg)
   virtual BOOL OnInitDialog();
        virtual void OnOK();
        afx_msg void OnWholeimage();
        afx_msg void OnSelchangeThreshSelector();
        afx_msg void OnSelchangeWaveletType();
        //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif



