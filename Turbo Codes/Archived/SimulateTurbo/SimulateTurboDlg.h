// SimulateTurboDlg.h : header file
//

#if !defined(AFX_SIMULATETURBODLG_H__95AF69E5_553A_4A2C_AA96_73EEBF16D1CB__INCLUDED_)
#define AFX_SIMULATETURBODLG_H__95AF69E5_553A_4A2C_AA96_73EEBF16D1CB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "montecarlo.h"
#include "interleaver.h"
#include "vector.h"
#include "matrix.h"
#include "timer.h"
#include <math.h>
#include <string.h>
#include "WorkerThread.h"
#include "StringStreamBuf.h"

typedef struct {
	int	nConfidence;
	int	nAccuracy;

	int	nChannel;
	int	nModulation;

   int	nIterations;
	bool  bFast;

	CString sCodec;

   vector<double> vdSNR;
   vector<int>    viSamples;
   matrix<double> mdEstimate;
   matrix<double> mdError;

	timer tSetup;
	timer tSimulation;
	bool bPresent;

   CString sFilename;
} STurboResults;

/////////////////////////////////////////////////////////////////////////////
// CSimulateTurboDlg dialog

class CSimulateTurboDlg : public CDialog, private montecarlo, CWorkerThread
{
// Construction
public:
	CSimulateTurboDlg(CWnd* pParent = NULL);	// standard constructor
	void ThreadProc();

// Dialog Data
	//{{AFX_DATA(CSimulateTurboDlg)
	enum { IDD = IDD_SIMULATETURBO_DIALOG };
	CProgressCtrl	m_pcTotal;
	CProgressCtrl	m_pcCurrent;
	int		m_nAccuracy;
	int		m_nConfidence;
	double	m_dSNRmax;
	double	m_dSNRmin;
	double	m_dSNRstep;
	int		m_nIterations;
	BOOL	m_bParallel;
	int		m_nModulation;
	int		m_nCodeType;
	int		m_nInputs;
	int		m_nOutputs;
	CString	m_sGenerators;
	int		m_nMemory;
	int		m_nSets;
	BOOL	m_bFast;
	int		m_nPuncturing;
	int		m_nTau;
	int		m_nChannel;
	CString	m_sPathName;
	BOOL	m_bEndAtZero;
	BOOL	m_bTermJPL;
	BOOL	m_bTermSimile;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CSimulateTurboDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	bool interrupt();
	void UpdateButtons(const bool bWorking);
	void ResetDisplay();
	void display(const int pass, const double cur_accuracy, const double cur_mean);
	HICON m_hIcon;

   vector<interleaver *> m_vpInterleavers;
   STurboResults m_sResults;

	// Generated message map functions
	//{{AFX_MSG(CSimulateTurboDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnLoad();
	afx_msg void OnStart();
	afx_msg void OnStop();
	afx_msg void OnSave();
	virtual void OnOK();
	virtual void OnCancel();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SIMULATETURBODLG_H__95AF69E5_553A_4A2C_AA96_73EEBF16D1CB__INCLUDED_)
