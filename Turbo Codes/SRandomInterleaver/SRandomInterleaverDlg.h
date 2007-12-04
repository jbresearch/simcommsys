// SRandomInterleaverDlg.h : header file
//

#if !defined(AFX_SRANDOMINTERLEAVERDLG_H__3A48070D_69EE_4E9F_A7FF_FDE74E4D2635__INCLUDED_)
#define AFX_SRANDOMINTERLEAVERDLG_H__3A48070D_69EE_4E9F_A7FF_FDE74E4D2635__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "WorkerThread.h"
#include "timer.h"
#include "vector.h"

/////////////////////////////////////////////////////////////////////////////
// CSRandomInterleaverDlg dialog

class CSRandomInterleaverDlg : public CDialog, libwin::CWorkerThread
{
// Construction
public:
        void ThreadProc();
        CSRandomInterleaverDlg(CWnd* pParent = NULL);   // standard constructor
   ~CSRandomInterleaverDlg();

// Dialog Data
        //{{AFX_DATA(CSRandomInterleaverDlg)
        enum { IDD = IDD_SRANDOMINTERLEAVER_DIALOG };
        CProgressCtrl   m_pcAttempt;
        CProgressCtrl   m_pcProgress;
        int             m_nSpread;
        int             m_nTau;
        int             m_nSeed;
        int             m_nAttempts;
        int             m_nUsedSeed;
        //}}AFX_DATA

        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSRandomInterleaverDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);        // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:
        int m_nUsedSpread;
        bool m_bValidResults;
        void UpdateButtons(const bool bWorking);
        HICON m_hIcon;
   libbase::vector<int> m_viInterleaver;
   libbase::timer m_tDuration;

        // Generated message map functions
        //{{AFX_MSG(CSRandomInterleaverDlg)
        virtual BOOL OnInitDialog();
        afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
        afx_msg void OnPaint();
        afx_msg HCURSOR OnQueryDragIcon();
        afx_msg void OnSuggest();
        afx_msg void OnSave();
        afx_msg void OnStart();
        afx_msg void OnStop();
        virtual void OnOK();
        virtual void OnCancel();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SRANDOMINTERLEAVERDLG_H__3A48070D_69EE_4E9F_A7FF_FDE74E4D2635__INCLUDED_)
