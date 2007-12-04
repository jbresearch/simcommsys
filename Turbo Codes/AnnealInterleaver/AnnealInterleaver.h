// AnnealInterleaver.h : main header file for the ANNEALINTERLEAVER application
//

#if !defined(AFX_ANNEALINTERLEAVER_H__3A206D01_0F0F_408B_98BB_D18C92A1055A__INCLUDED_)
#define AFX_ANNEALINTERLEAVER_H__3A206D01_0F0F_408B_98BB_D18C92A1055A__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
        #error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"           // main symbols
#include "RoutedIO.h"

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverApp:
// See AnnealInterleaver.cpp for the implementation of this class
//

/*
  Version 1.01 (9 Oct 2006)
  modified redirection of cerr/clog/cout to used read-buffer manipulation instead of
  direct assignment. This was necessitated in VS .NET 2005, but the code was not
  written to be compiler-dependent in the hope that this should still work on older
  Visual compilers.

  Version 1.02 (10 Nov 2006)
  * updated to use library namespaces.
  * made class a derivative of CRoutedIO.
  
  Version 1.03 (28 Nov 2007)
  * modifications to silence 64-bit portability warnings
    - changed response type from int to INT_PTR in InitInstance()
*/

class CAnnealInterleaverApp : public CWinApp, libwin::CRoutedIO
{
public:
        CAnnealInterleaverApp();

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CAnnealInterleaverApp)
        public:
        virtual BOOL InitInstance();
        //}}AFX_VIRTUAL

// Implementation

        //{{AFX_MSG(CAnnealInterleaverApp)
                // NOTE - the ClassWizard will add and remove member functions here.
                //    DO NOT EDIT what you see in these blocks of generated code !
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ANNEALINTERLEAVER_H__3A206D01_0F0F_408B_98BB_D18C92A1055A__INCLUDED_)
