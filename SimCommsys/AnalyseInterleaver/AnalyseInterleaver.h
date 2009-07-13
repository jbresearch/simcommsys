// AnalyseInterleaver.h : main header file for the ANALYSEINTERLEAVER application
//

#if !defined(AFX_ANALYSEINTERLEAVER_H__89E75CC4_1552_4D3F_8893_EB326FAB5603__INCLUDED_)
#define AFX_ANALYSEINTERLEAVER_H__89E75CC4_1552_4D3F_8893_EB326FAB5603__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
        #error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"           // main symbols
#include "RoutedIO.h"

/////////////////////////////////////////////////////////////////////////////
// CAnalyseInterleaverApp:
// See AnalyseInterleaver.cpp for the implementation of this class
//

/*
   \version 1.01 (9 Oct 2006)
  modified redirection of cerr/clog/cout to used read-buffer manipulation instead of
  direct assignment. This was necessitated in VS .NET 2005, but the code was not
  written to be compiler-dependent in the hope that this should still work on older
  Visual compilers.

   \version 1.02 (10 Nov 2006)
   - updated to use library namespaces.
   - made class a derivative of CRoutedIO.

   \version 1.03 (28 Nov 2007)
   - modifications to silence 64-bit portability warnings
    - changed response type from int to INT_PTR in InitInstance()
*/

class CAnalyseInterleaverApp : public CWinApp, libwin::CRoutedIO
{
public:
        CAnalyseInterleaverApp();

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CAnalyseInterleaverApp)
        public:
        virtual BOOL InitInstance();
        //}}AFX_VIRTUAL

// Implementation

        //{{AFX_MSG(CAnalyseInterleaverApp)
                // NOTE - the ClassWizard will add and remove member functions here.
                //    DO NOT EDIT what you see in these blocks of generated code !
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ANALYSEINTERLEAVER_H__89E75CC4_1552_4D3F_8893_EB326FAB5603__INCLUDED_)
