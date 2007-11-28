// SRandomInterleaver.h : main header file for the SRANDOMINTERLEAVER application
//

#if !defined(AFX_SRANDOMINTERLEAVER_H__9AC13527_8ED1_4C5F_BE07_4E7B7619DA29__INCLUDED_)
#define AFX_SRANDOMINTERLEAVER_H__9AC13527_8ED1_4C5F_BE07_4E7B7619DA29__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols
#include "RoutedIO.h"

/////////////////////////////////////////////////////////////////////////////
// CSRandomInterleaverApp:
// See SRandomInterleaver.cpp for the implementation of this class
//

/*
  Version 1.00 (undated)
  initial version

  Version 1.01 (22 Apr 2002)
  updated thread function so that progress indicator is only shown every 100ms, and
  not any faster - added this after I realised that most time was being spent in
  kernel functions.

  Version 1.02 (23 Apr 2002)
  added a box to display currently used seed; also updated the progress indicators to
  use the 32-bit range functions to allow seeds and tau values greater than 32k. Finally,
  updated the save function to suggest the filename based on the actual seed used, not
  the starting seed.

  Version 1.03 (9 Oct 2006)
  modified redirection of cerr/clog/cout to used read-buffer manipulation instead of
  direct assignment. This was necessitated in VS .NET 2005, but the code was not
  written to be compiler-dependent in the hope that this should still work on older
  Visual compilers.

  Version 1.04 (10 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
  * made class a derivative of CRoutedIO.
  
  Version 1.05 (28 Nov 2007)
  * modifications to silence 64-bit portability warnings
    - changed response type from int to INT_PTR in InitInstance()
*/
class CSRandomInterleaverApp : public CWinApp, libwin::CRoutedIO
{
public:
	CSRandomInterleaverApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CSRandomInterleaverApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation

	//{{AFX_MSG(CSRandomInterleaverApp)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SRANDOMINTERLEAVER_H__9AC13527_8ED1_4C5F_BE07_4E7B7619DA29__INCLUDED_)
