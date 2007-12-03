// SimulateCommsysView.h : interface of the CSimulateCommsysView class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_SIMULATECOMMSYSVIEW_H__F7883806_938E_47AB_AEAD_6212F56B6C95__INCLUDED_)
#define AFX_SIMULATECOMMSYSVIEW_H__F7883806_938E_47AB_AEAD_6212F56B6C95__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "montecarlo.h"
#include "WorkerThread.h"

class CSimulateCommsysView : public CListView, private libcomm::montecarlo, libwin::CWorkerThread
{
protected: // create from serialization only
        CSimulateCommsysView();
        DECLARE_DYNCREATE(CSimulateCommsysView)

// Attributes
public:
        CSimulateCommsysDoc* GetDocument();

// Operations
public:

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSimulateCommsysView)
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
        virtual ~CSimulateCommsysView();
#ifdef _DEBUG
        virtual void AssertValid() const;
        virtual void Dump(CDumpContext& dc) const;
#endif
   bool Working() const { return ThreadWorking(); };
   double GetSNR() const { return m_dSNR; };
   double GetProgress() const { return m_dProgress; };

protected:
   double m_dSNRmin, m_dSNRmax, m_dSNRstep;
   double m_dSNR, m_dProgress;

   // internal functions
   void InsertResults(const double dSNR, const int iSamples, const libbase::vector<double>& vdEstimate, const libbase::vector<double>& vdError, const double dElapsed);
   // montecarlo overrides
   bool interrupt() { return ThreadInterrupted(); };
        void display(const int pass, const double cur_accuracy, const double cur_mean);
   // WorkerThread overrides
        void ThreadProc();

// Generated message map functions
protected:
        //{{AFX_MSG(CSimulateCommsysView)
        afx_msg void OnSystemChannel();
        afx_msg void OnSystemModulation();
        afx_msg void OnUpdateSystemChannel(CCmdUI* pCmdUI);
        afx_msg void OnUpdateSystemModulation(CCmdUI* pCmdUI);
        afx_msg void OnUpdateSystemPuncturing(CCmdUI* pCmdUI);
        afx_msg void OnUpdateSystemCodec(CCmdUI* pCmdUI);
        afx_msg void OnSystemPuncturing();
        afx_msg void OnSystemCodec();
        afx_msg void OnSimulationAccuracy();
        afx_msg void OnSimulationRange();
        afx_msg void OnSimulationStart();
        afx_msg void OnSimulationStop();
        afx_msg void OnUpdateSimulationStart(CCmdUI* pCmdUI);
        afx_msg void OnUpdateSimulationStop(CCmdUI* pCmdUI);
        afx_msg void OnUpdateSimulationAccuracy(CCmdUI* pCmdUI);
        afx_msg void OnUpdateSimulationRange(CCmdUI* pCmdUI);
        afx_msg void OnUpdateFileSave(CCmdUI* pCmdUI);
        afx_msg void OnUpdateFileSaveAs(CCmdUI* pCmdUI);
        afx_msg void OnSimulationBenchmark();
        afx_msg void OnUpdateSimulationBenchmark(CCmdUI* pCmdUI);
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in SimulateCommsysView.cpp
inline CSimulateCommsysDoc* CSimulateCommsysView::GetDocument()
   { return (CSimulateCommsysDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SIMULATECOMMSYSVIEW_H__F7883806_938E_47AB_AEAD_6212F56B6C95__INCLUDED_)
