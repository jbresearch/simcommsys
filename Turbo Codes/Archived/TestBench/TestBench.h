// TestBench.h : main header file for the TESTBENCH application
//

#if !defined(AFX_TESTBENCH_H__7987393F_7990_4AF4_88EF_A14C3C4671D1__INCLUDED_)
#define AFX_TESTBENCH_H__7987393F_7990_4AF4_88EF_A14C3C4671D1__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols

#include "TraceStreamBuf.h"
#include "MBoxStreamBuf.h"

/////////////////////////////////////////////////////////////////////////////
// CTestBenchApp:
// See TestBench.cpp for the implementation of this class
//

/*
  Version 1.01 (9 Oct 2006)
  modified redirection of cerr/clog/cout to used read-buffer manipulation instead of
  direct assignment. This was necessitated in VS .NET 2005, but the code was not
  written to be compiler-dependent in the hope that this should still work on older
  Visual compilers.
*/

class CTestBenchApp : public CWinApp
{
private:
   CTraceStreamBuf m_tracer;
   CMBoxStreamBuf m_msgbox;

public:
	CTestBenchApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CTestBenchApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation

	//{{AFX_MSG(CTestBenchApp)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TESTBENCH_H__7987393F_7990_4AF4_88EF_A14C3C4671D1__INCLUDED_)
