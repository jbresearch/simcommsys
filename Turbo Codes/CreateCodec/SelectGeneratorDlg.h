#if !defined(AFX_SELECTGENERATORDLG_H__547ACC66_C792_4DDD_89B5_2426FACD21D5__INCLUDED_)
#define AFX_SELECTGENERATORDLG_H__547ACC66_C792_4DDD_89B5_2426FACD21D5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectGeneratorDlg.h : header file
//

#include "matrix.h"
#include "bitfield.h"

/////////////////////////////////////////////////////////////////////////////
// CSelectGeneratorDlg dialog

class CSelectGeneratorDlg : public CDialog
{
// Construction
public:
	CSelectGeneratorDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CSelectGeneratorDlg)
	enum { IDD = IDD_GENERATOR };
	CSliderCtrl	m_scOutput;
	CSliderCtrl	m_scInput;
	CString	m_sValue;
	//}}AFX_DATA
   libbase::matrix<libbase::bitfield> m_mbGenerator;

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CSelectGeneratorDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	void UpdateValue();

	// Generated message map functions
	//{{AFX_MSG(CSelectGeneratorDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnReleasedcaptureInput(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnReleasedcaptureOutput(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnUpdateValue();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTGENERATORDLG_H__547ACC66_C792_4DDD_89B5_2426FACD21D5__INCLUDED_)
