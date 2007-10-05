// TestBenchDlg.h : header file
//

#if !defined(AFX_TESTBENCHDLG_H__5C15E067_22A2_439B_AB3B_1234460F6FE0__INCLUDED_)
#define AFX_TESTBENCHDLG_H__5C15E067_22A2_439B_AB3B_1234460F6FE0__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "interleaver.h"

/////////////////////////////////////////////////////////////////////////////
// CTestBenchDlg dialog

class CTestBenchDlg : public CDialog
{
// Construction
public:
	CTestBenchDlg(CWnd* pParent = NULL);	// standard constructor
	virtual ~CTestBenchDlg();	// standard destructor

// Dialog Data
	//{{AFX_DATA(CTestBenchDlg)
	enum { IDD = IDD_TESTBENCH_DIALOG };
	int		m_nType;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CTestBenchDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	interleaver *pInter;
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CTestBenchDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnLoad();
	afx_msg void OnSave();
	afx_msg void OnMake();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TESTBENCHDLG_H__5C15E067_22A2_439B_AB3B_1234460F6FE0__INCLUDED_)
