// SimulateCommsys.h : main header file for the SIMULATECOMMSYS application
//

#if !defined(AFX_SIMULATECOMMSYS_H__2EBC5390_B306_42A4_9153_396FEFCDDE06__INCLUDED_)
#define AFX_SIMULATECOMMSYS_H__2EBC5390_B306_42A4_9153_396FEFCDDE06__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
        #error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"       // main symbols
#include "RoutedIO.h"

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysApp:
// See SimulateCommsys.cpp for the implementation of this class
//

/*
   \version 1.10 (18 Apr 2002)
  modified Benchmark dialog - now uses a worker thread to do the benchmark. Note that
  worker thread operates at priority_normal (rather than the default _lowest) to ensure
  good results while not impacting responsiveness too much.
  also made the default "document" to have an AWGN channel and BPSK modulation, rather
  than nothing.

   \version 1.11 (9 Oct 2006)
  modified redirection of cerr/clog/cout to used read-buffer manipulation instead of
  direct assignment. This was necessitated in VS .NET 2005, but the code was not
  written to be compiler-dependent in the hope that this should still work on older
  Visual compilers.

   \version 1.12 (10 Nov 2006)
   - updated to use library namespaces.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.
   - made class a derivative of CRoutedIO.

   \version 1.13 (18 Dec 2007)
   - updated BenchmarkDlg so that call to experiment::sample is only for a single frame.
*/

class CSimulateCommsysApp : public CWinApp, libwin::CRoutedIO
{
public:
        CSimulateCommsysApp();

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSimulateCommsysApp)
        public:
        virtual BOOL InitInstance();
        //}}AFX_VIRTUAL

// Implementation
        //{{AFX_MSG(CSimulateCommsysApp)
        afx_msg void OnAppAbout();
                // NOTE - the ClassWizard will add and remove member functions here.
                //    DO NOT EDIT what you see in these blocks of generated code !
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SIMULATECOMMSYS_H__2EBC5390_B306_42A4_9153_396FEFCDDE06__INCLUDED_)
