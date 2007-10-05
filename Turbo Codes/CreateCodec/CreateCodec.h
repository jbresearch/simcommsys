// CreateCodec.h : main header file for the CREATECODEC application
//

#if !defined(AFX_CREATECODEC_H__B5C008F2_193C_4096_88E0_2FB033712D12__INCLUDED_)
#define AFX_CREATECODEC_H__B5C008F2_193C_4096_88E0_2FB033712D12__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"       // main symbols

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecApp:
// See CreateCodec.cpp for the implementation of this class
//

/*
  Version 1.10 (10 Nov 2006)
  * updated to use library namespaces.
*/

class CCreateCodecApp : public CWinApp
{
public:
	CCreateCodecApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCreateCodecApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation
	//{{AFX_MSG(CCreateCodecApp)
	afx_msg void OnAppAbout();
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CREATECODEC_H__B5C008F2_193C_4096_88E0_2FB033712D12__INCLUDED_)
