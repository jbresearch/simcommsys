#ifndef __histogram2d_h
#define __histogram2d_h

#include "matrix.h"

/*
  Version 1.10 (6 Nov 2006)
  * defined class and associated data within "libwin" namespace.
  * removed pragma once directive, as this is unnecessary
  * changed unique define to conform with that used in other libraries
  
  Version 1.11 (28 Nov 2007)
  * modifications to silence 64-bit portability warnings
    - changed type cast from int to UINT_PTR in UpdateData()
*/

namespace libwin {

class CHistogram2D : public CWnd
{
// Construction
public:
        CHistogram2D();

// Attributes
public:

// Operations
public:

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CHistogram2D)
        //}}AFX_VIRTUAL

// Implementation
public:
        virtual ~CHistogram2D();
        static bool RegisterWndClass(HINSTANCE hInstance);
   static void UpdateData(CWnd* pWnd, libbase::matrix<int>& m);

        // Generated message map functions
protected:
        libbase::matrix<int> m_data;
   int m_maxval;
        //{{AFX_MSG(CHistogram2D)
        afx_msg void OnPaint();
        //}}AFX_MSG
   afx_msg void OnUser(WPARAM wParam, LPARAM lParam);
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

}; // end namespace

#endif
