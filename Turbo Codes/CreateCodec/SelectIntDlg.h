#if !defined(AFX_SELECTINTDLG_H__992CB40F_9697_4CCF_990A_400BDA1DBC23__INCLUDED_)
#define AFX_SELECTINTDLG_H__992CB40F_9697_4CCF_990A_400BDA1DBC23__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectIntDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectIntDlg dialog

class CSelectIntDlg : public CDialog
{
// Construction
public:
	CSelectIntDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CSelectIntDlg)
	enum { IDD = IDD_INT };
	int		m_nValue;
	//}}AFX_DATA


// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CSelectIntDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:

	// Generated message map functions
	//{{AFX_MSG(CSelectIntDlg)
		// NOTE: the ClassWizard will add member functions here
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTINTDLG_H__992CB40F_9697_4CCF_990A_400BDA1DBC23__INCLUDED_)
