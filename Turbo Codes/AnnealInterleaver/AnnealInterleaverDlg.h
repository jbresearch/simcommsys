// AnnealInterleaverDlg.h : header file
//

#if !defined(AFX_ANNEALINTERLEAVERDLG_H__B8D362B1_70F2_4117_B04D_3885D130024F__INCLUDED_)
#define AFX_ANNEALINTERLEAVERDLG_H__B8D362B1_70F2_4117_B04D_3885D130024F__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "anneal_interleaver.h"
#include "annealer.h"
#include "timer.h"
#include "WorkerThread.h"

#include "afxmt.h"
#include "wmdefines.h"

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverDlg dialog

class CAnnealInterleaverDlg : public CDialog, private libcomm::annealer, libwin::CWorkerThread
{
// Construction
public:
        void ResetDisplay();
        void UpdateButtons(const bool bWorking);
        CAnnealInterleaverDlg(CWnd* pParent = NULL);    // standard constructor

// Dialog Data
        //{{AFX_DATA(CAnnealInterleaverDlg)
        enum { IDD = IDD_ANNEALINTERLEAVER_DIALOG };
        CProgressCtrl   m_pcProgress;
        int             m_nM;
        int             m_nSeed;
        int             m_nSets;
        int             m_nTau;
        BOOL    m_bTerm;
        int             m_nType;
        int             m_nMinChanges;
        int             m_nMinIter;
        double  m_dFinalTemp;
        double  m_dInitTemp;
        double  m_dRate;
        //}}AFX_DATA

        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CAnnealInterleaverDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);        // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:
        void ThreadProc();
        HICON m_hIcon;
   libbase::timer m_tSetup;
   libbase::timer m_tSimulation;
        bool m_bSystemPresent;
        libcomm::anneal_interleaver *m_system;

   double m_dTemp;
   double m_dMean;
   double m_dSigma;
   double m_dHi;
   double m_dLo;
   double m_dPercent;

   bool interrupt();
        void display(const double T, const double percent, const libbase::rvstatistics E);

        // Generated message map functions
        //{{AFX_MSG(CAnnealInterleaverDlg)
        virtual BOOL OnInitDialog();
        afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
        afx_msg void OnPaint();
        afx_msg HCURSOR OnQueryDragIcon();
        afx_msg void OnStart();
        virtual void OnOK();
        virtual void OnCancel();
        afx_msg void OnStop();
        afx_msg void OnSave();
        afx_msg void OnTimer(UINT nIDEvent);
        afx_msg void OnSuspend();
        afx_msg void OnResume();
        //}}AFX_MSG
   afx_msg LONG OnThreadDisplay(WPARAM wParam, LPARAM lParam);
   afx_msg LONG OnThreadFinish(WPARAM wParam, LPARAM lParam);
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ANNEALINTERLEAVERDLG_H__B8D362B1_70F2_4117_B04D_3885D130024F__INCLUDED_)
