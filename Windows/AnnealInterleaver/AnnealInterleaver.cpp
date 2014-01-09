/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

// AnnealInterleaver.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "AnnealInterleaver.h"
#include "AnnealInterleaverDlg.h"
#include "StatusGraph.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverApp

BEGIN_MESSAGE_MAP(CAnnealInterleaverApp, CWinApp)
        //{{AFX_MSG_MAP(CAnnealInterleaverApp)
                // NOTE - the ClassWizard will add and remove mapping macros here.
                //    DO NOT EDIT what you see in these blocks of generated code!
        //}}AFX_MSG
        ON_COMMAND(ID_HELP, CWinApp::OnHelp)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverApp construction

CAnnealInterleaverApp::CAnnealInterleaverApp()
   {
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CAnnealInterleaverApp object

CAnnealInterleaverApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverApp initialization

BOOL CAnnealInterleaverApp::InitInstance()
{
        // Standard initialization
        // If you are not using these features and wish to reduce the size
        //  of your final executable, you should remove from the following
        //  the specific initialization routines you do not need.

#if _MSC_VER < 1400
   #ifdef _AFXDLL
           Enable3dControls();                  // Call this when using MFC in a shared DLL
   #else
           Enable3dControlsStatic();    // Call this when linking to MFC statically
   #endif
#endif

   // Enable Custom Controls
   libwin::CStatusGraph::RegisterWndClass(AfxGetInstanceHandle());

   CAnnealInterleaverDlg dlg;
        m_pMainWnd = &dlg;
        INT_PTR nResponse = dlg.DoModal();

        // Since the dialog has been closed, return FALSE so that we exit the
        //  application, rather than start the application's message pump.
        return FALSE;
}
