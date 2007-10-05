// SimulateTurbo.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "SimulateTurbo.h"
#include "SimulateTurboDlg.h"
#include "StatusGraph.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSimulateTurboApp

BEGIN_MESSAGE_MAP(CSimulateTurboApp, CWinApp)
	//{{AFX_MSG_MAP(CSimulateTurboApp)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG
	ON_COMMAND(ID_HELP, CWinApp::OnHelp)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSimulateTurboApp construction

CSimulateTurboApp::CSimulateTurboApp()
{
   ostream tracer(&m_tracer);
   ostream msgbox(&m_msgbox);
   clog.rdbuf(tracer.rdbuf());
   cout.rdbuf(tracer.rdbuf());
   cerr.rdbuf(msgbox.rdbuf());
}

/////////////////////////////////////////////////////////////////////////////
// The one and only CSimulateTurboApp object

CSimulateTurboApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CSimulateTurboApp initialization

BOOL CSimulateTurboApp::InitInstance()
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

   // Enable Custom Controls
   CStatusGraph::RegisterWndClass(AfxGetInstanceHandle());

	CSimulateTurboDlg dlg;
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
