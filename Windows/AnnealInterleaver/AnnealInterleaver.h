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
