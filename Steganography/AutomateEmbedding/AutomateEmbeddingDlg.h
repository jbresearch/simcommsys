#ifndef afx_automateembeddingdlg_h
#define afx_automateembeddingdlg_h

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
// CAutomateEmbeddingDlg dialog

class CAutomateEmbeddingDlg : public CDialog
{
// Construction
public:
   CAutomateEmbeddingDlg(CWnd* pParent = NULL);   // standard constructor

   libwin::CPSAutomate*  m_pPSAutomate;

// Dialog Data
   //{{AFX_DATA(CAutomateEmbeddingDlg)
	enum { IDD = IDD_DIALOG1 };
	int		m_nJpegMin;
	int		m_nJpegMax;
	double	m_dStrengthMax;
	double	m_dStrengthMin;
	BOOL	m_bJpeg;
	int		m_nJpegStep;
	double	m_dStrengthStep;
	CString	m_sOutput;
	//}}AFX_DATA

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAutomateEmbeddingDlg)
   protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

// Implementation
protected:

   // Generated message map functions
   //{{AFX_MSG(CAutomateEmbeddingDlg)
   virtual BOOL OnInitDialog();
	afx_msg void OnJpeg();
	virtual void OnOK();
	afx_msg void OnOutputBrowse();
	//}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif 


