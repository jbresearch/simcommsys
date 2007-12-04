// BenchmarkDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SimulateCommsys.h"
#include "BenchmarkDlg.h"

#include "randgen.h"
#include "commsys.h"
#include "timer.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CBenchmarkDlg dialog


CBenchmarkDlg::CBenchmarkDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CBenchmarkDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CBenchmarkDlg)
        m_dSNR = 0.0;
        m_dTime = 0.0;
        m_sBER = _T("");
        m_sChannel = _T("");
        m_sElapsed = _T("");
        m_sFrames = _T("");
        m_sModulator = _T("");
        m_sPuncture = _T("");
        m_sSpeed = _T("");
        m_sCodec = _T("");
        //}}AFX_DATA_INIT
}


void CBenchmarkDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CBenchmarkDlg)
        DDX_Control(pDX, IDC_PROGRESS, m_pcProgress);
        DDX_Text(pDX, IDC_SNR, m_dSNR);
        DDX_Text(pDX, IDC_TIME, m_dTime);
        DDX_Text(pDX, IDC_BER, m_sBER);
        DDX_Text(pDX, IDC_CHANNEL, m_sChannel);
        DDX_Text(pDX, IDC_ELAPSED, m_sElapsed);
        DDX_Text(pDX, IDC_FRAMES, m_sFrames);
        DDX_Text(pDX, IDC_MODULATOR, m_sModulator);
        DDX_Text(pDX, IDC_PUNCTURE, m_sPuncture);
        DDX_Text(pDX, IDC_SPEED, m_sSpeed);
        DDX_Text(pDX, IDC_CODEC, m_sCodec);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CBenchmarkDlg, CDialog)
        //{{AFX_MSG_MAP(CBenchmarkDlg)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CBenchmarkDlg message handlers

BOOL CBenchmarkDlg::OnInitDialog() 
   {
   CDialog::OnInitDialog();
   
   // TODO: Add extra initialization here
   if(m_pCodec != NULL)
      m_sCodec = m_pCodec->description().c_str();
   if(m_pPuncture != NULL)
      m_sPuncture = m_pPuncture->description().c_str();
   if(m_pModulator != NULL)
      m_sModulator = m_pModulator->description().c_str();
   if(m_pChannel != NULL)
      m_sChannel = m_pChannel->description().c_str();
   UpdateData(false);

   m_pcProgress.SetRange(0, 100);
   
   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CBenchmarkDlg::OnOK() 
   {
   UpdateData(true);
   ASSERT(m_pCodec != NULL);
   ASSERT(m_pModulator != NULL);
   ASSERT(m_pChannel != NULL);

   // Disable further input from user
   GetDlgItem(IDC_TIME)->EnableWindow(false);
   GetDlgItem(IDC_SNR)->EnableWindow(false);
   GetDlgItem(IDOK)->EnableWindow(false);
   GetDlgItem(IDCANCEL)->EnableWindow(false);

   ThreadStart(THREAD_PRIORITY_NORMAL);
   }

void CBenchmarkDlg::OnCancel() 
   {
   CDialog::OnCancel();
   }

void CBenchmarkDlg::ThreadProc() 
   {
   // Source Generator
   libbase::randgen src;
   // The complete communication system
   libcomm::commsys system(&src, m_pCodec, m_pModulator, m_pPuncture, m_pChannel);

   // Work out at the SNR value required
   m_pChannel->set_snr(m_dSNR);

   // Prepare for simulation run
   const int count = system.count();
   libbase::vector<double> est(count), sum(count);
   sum = 0;
   int frames = 0, passes = 0;
   system.seed(0);

   // Time the simulation
   libbase::timer main_timer;
   main_timer.start();
   while(main_timer.elapsed() < m_dTime)
      {
      system.sample(est, frames);
      sum += est;
      passes++;
      m_pcProgress.SetPos(int(100*main_timer.elapsed()/m_dTime));
      }
   main_timer.stop();

   // Work out averages
   sum /= double(passes);
   // Print results (for confirming accuracy)
   m_sBER.Format("%0.4g", sum(count-2));

   // Output timing statistics
   m_sElapsed = std::string(main_timer).c_str();
   m_sFrames.Format("%d (%d passes)", frames, passes);
   m_sSpeed.Format("%4.2f frames/CPUsec", frames/(main_timer.elapsed()*main_timer.usage()/100));
   UpdateData(false);

   // Enable input again
   GetDlgItem(IDC_TIME)->EnableWindow(true);
   GetDlgItem(IDC_SNR)->EnableWindow(true);
   GetDlgItem(IDOK)->EnableWindow(true);
   GetDlgItem(IDCANCEL)->EnableWindow(true);
   }
