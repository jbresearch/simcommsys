#ifndef afx_pluginwizarddlg_h
#define afx_pluginwizarddlg_h

/////////////////////////////////////////////////////////////////////////////
// CPluginWizardDlg dialog

class CPluginWizardDlg : public CDialog
{
// Construction
public:
	CString m_sPath;
	CPluginWizardDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CPluginWizardDlg)
	enum { IDD = IDD_PLUGINWIZARD_DIALOG };
	int		m_nType;
	CString	m_sNewName;
	CString	m_sOldName;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CPluginWizardDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CPluginWizardDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	virtual void OnOK();
	afx_msg void OnSelchangeType();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
