#ifndef __statusgraph_h
#define __statusgraph_h

#include "afxtempl.h"
#include "wmdefines.h"

/////////////////////////////////////////////////////////////////////////////
// CStatusGraph window

/*
  Version 1.10 (26 Feb 2002)
  noticed that the user messages were interfering with those of other classes. Started
  keeping track of user message allocation in a separate file in LibWin.

  Version 1.11 (7 Oct 2006)
  changed include fstream.h to fstream, and added usage of namespace std, to conform
  with modern library usage.

  Version 1.20 (6 Nov 2006)
  * defined class and associated data within "libwin" namespace.
  * removed pragma once directive, as this is unnecessary
  * changed unique define to conform with that used in other libraries
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
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

}; // end namespace

#endif
