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

#ifndef __statusgraph_h
#define __statusgraph_h

#include "afxtempl.h"
#include "wmdefines.h"

/////////////////////////////////////////////////////////////////////////////
// CStatusGraph window

/*
   \version 1.10 (26 Feb 2002)
  noticed that the user messages were interfering with those of other classes. Started
  keeping track of user message allocation in a separate file in LibWin.

   \version 1.11 (7 Oct 2006)
  changed include fstream.h to fstream, and added usage of namespace std, to conform
  with modern library usage.

   \version 1.20 (6 Nov 2006)
   - defined class and associated data within "libwin" namespace.
   - removed pragma once directive, as this is unnecessary
   - changed unique define to conform with that used in other libraries
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.21 (28 Nov 2007)
   - modifications to silence 64-bit portability warnings
    - changed type cast from int to UINT_PTR in Insert()
*/

namespace libwin {

class CStatusGraph : public CWnd
{
// Construction
public:
        CStatusGraph();

// Attributes
public:

// Operations
public:

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CStatusGraph)
        //}}AFX_VIRTUAL

// Implementation
public:
        virtual ~CStatusGraph();
        static bool RegisterWndClass(HINSTANCE hInstance);
        static void Reset(CWnd* pWnd);
        static void Insert(CWnd* pWnd, const double x);

        // Generated message map functions
protected:
   CList<double,double> m_data;
   double m_maxval;
        //{{AFX_MSG(CStatusGraph)
        afx_msg void OnPaint();
   afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
        //}}AFX_MSG
   afx_msg void OnReset(WPARAM wParam, LPARAM lParam);
   afx_msg void OnInsert(WPARAM wParam, LPARAM lParam);
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

} // end namespace

#endif
