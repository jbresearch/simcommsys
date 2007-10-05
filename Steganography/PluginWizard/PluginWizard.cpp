#include "stdafx.h"
#include "PluginWizard.h"
#include "PluginWizardDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CPluginWizardApp

BEGIN_MESSAGE_MAP(CPluginWizardApp, CWinApp)
//{{AFX_MSG_MAP(CPluginWizardApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG
ON_COMMAND(ID_HELP, CWinApp::OnHelp)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CPluginWizardApp construction

CPluginWizardApp::CPluginWizardApp()
   {
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CPluginWizardApp object

CPluginWizardApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CPluginWizardApp initialization

BOOL CPluginWizardApp::InitInstance()
   {
   // Create the dialog box
   CPluginWizardDlg dlg;
   m_pMainWnd = &dlg;

   // Obtain the path to the final output
   CString sPath = m_pszHelpFilePath;
   dlg.m_sPath = sPath.Left(sPath.Find(m_pszExeName));

   int nResponse = dlg.DoModal();
   if (nResponse == IDOK)
      {
      }
   else if (nResponse == IDCANCEL)
      {
      }
   
   // Since the dialog has been closed, return FALSE so that we exit the
   //  application, rather than start the application's message pump.
   return FALSE;
   }
