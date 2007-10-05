// TestBench.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "TestBench.h"
#include "TestBenchDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CTestBenchApp

BEGIN_MESSAGE_MAP(CTestBenchApp, CWinApp)
	//{{AFX_MSG_MAP(CTestBenchApp)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG
	ON_COMMAND(ID_HELP, CWinApp::OnHelp)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CTestBenchApp construction

CTestBenchApp::CTestBenchApp()
   {
   ostream tracer(&m_tracer);
   ostream msgbox(&m_msgbox);
   clog.rdbuf(tracer.rdbuf());
   cout.rdbuf(tracer.rdbuf());
   cerr.rdbuf(msgbox.rdbuf());
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CTestBenchApp object

CTestBenchApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CTestBenchApp initialization

BOOL CTestBenchApp::InitInstance()
{
	// Standard initialization
	// If you are not using these features and wish to reduce the size
	//  of your final executable, you should remove from the following
	//  the specific initialization routines you do not need.

#if _MSC_VER < 1400
   #ifdef _AFXDLL
	   Enable3dControls();			// Call this when using MFC in a shared DLL
   #else
	   Enable3dControlsStatic();	// Call this when linking to MFC statically
   #endif
#endif

	CTestBenchDlg dlg;
	m_pMainWnd = &dlg;
	int nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO: Place code here to handle when the dialog is
		//  dismissed with OK
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO: Place code here to handle when the dialog is
		//  dismissed with Cancel
	}

	// Since the dialog has been closed, return FALSE so that we exit the
	//  application, rather than start the application's message pump.
	return FALSE;
}
