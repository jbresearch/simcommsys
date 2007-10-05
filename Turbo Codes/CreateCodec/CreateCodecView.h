// CreateCodecView.h : interface of the CCreateCodecView class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_CREATECODECVIEW_H__9D4C3389_7789_4AB3_A111_43D415FE57BB__INCLUDED_)
#define AFX_CREATECODECVIEW_H__9D4C3389_7789_4AB3_A111_43D415FE57BB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CCreateCodecView : public CTreeView
{
protected: // create from serialization only
	CCreateCodecView();
	DECLARE_DYNCREATE(CCreateCodecView)

// Attributes
public:
	CCreateCodecDoc* GetDocument();

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCreateCodecView)
	public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual void OnInitialUpdate(); // called first time after construct
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CCreateCodecView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:
	void SelectInterleaver(HTREEITEM hItem);
	void SelectGenerator(HTREEITEM hItem);
	void SelectEncoder(HTREEITEM hItem);
	void SelectBool(HTREEITEM hItem);
	void SelectInt(HTREEITEM hItem);
	void DeleteChildren(HTREEITEM hItem);
	void SelectCodec(HTREEITEM hItem);

// Generated message map functions
protected:
	//{{AFX_MSG(CCreateCodecView)
	afx_msg void OnDblclk(NMHDR* pNMHDR, LRESULT* pResult);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in CreateCodecView.cpp
inline CCreateCodecDoc* CCreateCodecView::GetDocument()
   { return (CCreateCodecDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CREATECODECVIEW_H__9D4C3389_7789_4AB3_A111_43D415FE57BB__INCLUDED_)
