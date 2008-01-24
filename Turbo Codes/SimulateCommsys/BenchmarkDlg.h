#if !defined(AFX_BENCHMARKDLG_H__A7CEFAB3_1184_4454_A7F0_959D4CD3AFC2__INCLUDED_)
#define AFX_BENCHMARKDLG_H__A7CEFAB3_1184_4454_A7F0_959D4CD3AFC2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// BenchmarkDlg.h : header file
//

#include "WorkerThread.h"

#include "channel.h"
#include "modulator.h"
#include "puncture.h"
#include "codec.h"

/////////////////////////////////////////////////////////////////////////////
// CBenchmarkDlg dialog

class CBenchmarkDlg : public CDialog, private libwin::CWorkerThread
{
// Construction
public:
        CBenchmarkDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
   //{{AFX_DATA(CBenchmarkDlg)
   enum { IDD = IDD_BENCHMARK };
   CProgressCtrl   m_pcProgress;
   double  m_dSNR;
   double  m_dTime;
   CString m_sBER;
   CString m_sChannel;
   CString m_sElapsed;
   CString m_sFrames;
   CString m_sModulator;
   CString m_sPuncture;
   CString m_sSpeed;
   CString m_sCodec;
   //}}AFX_DATA
   libcomm::codec *m_pCodec;
   libcomm::puncture *m_pPuncture;
   libcomm::modulator *m_pModulator;
   libcomm::channel<libcomm::sigspace> *m_pChannel;


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CBenchmarkDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:
   void ThreadProc();

        // Generated message map functions
        //{{AFX_MSG(CBenchmarkDlg)
        virtual void OnOK();
        virtual void OnCancel();
        virtual BOOL OnInitDialog();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_BENCHMARKDLG_H__A7CEFAB3_1184_4454_A7F0_959D4CD3AFC2__INCLUDED_)
