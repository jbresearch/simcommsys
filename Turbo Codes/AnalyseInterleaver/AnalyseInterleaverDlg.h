// AnalyseInterleaverDlg.h : header file
//

#if !defined(AFX_ANALYSEINTERLEAVERDLG_H__797804D0_A1D8_4400_B11C_1A298E94C19E__INCLUDED_)
#define AFX_ANALYSEINTERLEAVERDLG_H__797804D0_A1D8_4400_B11C_1A298E94C19E__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "interleaver.h"
#include "file_lut.h"
#include "stream_lut.h"
#include "matrix.h"
#include "vector.h"

/////////////////////////////////////////////////////////////////////////////
// CAnalyseInterleaverDlg dialog

class CAnalyseInterleaverDlg : public CDialog
{
// Construction
public:
	CAnalyseInterleaverDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CAnalyseInterleaverDlg)
	enum { IDD = IDD_ANALYSEINTERLEAVER_DIALOG };
	CProgressCtrl	m_pcProgress;
	CString	m_sPathName;
	int		m_nTau;
	int		m_nSpread;
	int		m_nMaxDist;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAnalyseInterleaverDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
   libcomm::interleaver* m_pInterleaver;
   libbase::matrix<int> m_miIOSS;
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CAnalyseInterleaverDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnLoad();
	afx_msg void OnAnalyse();
	virtual void OnOK();
	virtual void OnCancel();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ANALYSEINTERLEAVERDLG_H__797804D0_A1D8_4400_B11C_1A298E94C19E__INCLUDED_)
