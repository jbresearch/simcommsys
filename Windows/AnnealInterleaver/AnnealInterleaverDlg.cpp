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

// AnnealInterleaverDlg.cpp : implementation file
//

#include "stdafx.h"
#include "AnnealInterleaver.h"
#include "AnnealInterleaverDlg.h"
#include "StatusGraph.h"
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
        CAboutDlg();

// Dialog Data
        //{{AFX_DATA(CAboutDlg)
        enum { IDD = IDD_ABOUTBOX };
        //}}AFX_DATA

        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CAboutDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:
        //{{AFX_MSG(CAboutDlg)
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
        //{{AFX_DATA_INIT(CAboutDlg)
        //}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CAboutDlg)
        //}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
        //{{AFX_MSG_MAP(CAboutDlg)
                // No message handlers
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverDlg dialog

CAnnealInterleaverDlg::CAnnealInterleaverDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CAnnealInterleaverDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CAnnealInterleaverDlg)
        m_nM = 0;
        m_nSeed = 0;
        m_nSets = 0;
        m_nTau = 0;
        m_bTerm = FALSE;
        m_nType = -1;
        m_nMinChanges = 0;
        m_nMinIter = 0;
        m_dFinalTemp = 0.0;
        m_dInitTemp = 0.0;
        m_dRate = 0.0;
        //}}AFX_DATA_INIT
        // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
        m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CAnnealInterleaverDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CAnnealInterleaverDlg)
        DDX_Control(pDX, IDC_PROGRESS, m_pcProgress);
        DDX_Text(pDX, IDC_M, m_nM);
        DDX_Text(pDX, IDC_SEED, m_nSeed);
        DDX_Text(pDX, IDC_SETS, m_nSets);
        DDX_Text(pDX, IDC_TAU, m_nTau);
        DDX_Check(pDX, IDC_TERM, m_bTerm);
        DDX_CBIndex(pDX, IDC_TYPE, m_nType);
        DDX_Text(pDX, IDC_MINCHANGES, m_nMinChanges);
        DDX_Text(pDX, IDC_MINITER, m_nMinIter);
        DDX_Text(pDX, IDC_FINALTEMP, m_dFinalTemp);
        DDX_Text(pDX, IDC_INITTEMP, m_dInitTemp);
        DDX_Text(pDX, IDC_RATE, m_dRate);
        //}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAnnealInterleaverDlg, CDialog)
        //{{AFX_MSG_MAP(CAnnealInterleaverDlg)
        ON_WM_SYSCOMMAND()
        ON_WM_PAINT()
        ON_WM_QUERYDRAGICON()
        ON_BN_CLICKED(IDC_START, OnStart)
        ON_BN_CLICKED(IDC_STOP, OnStop)
        ON_BN_CLICKED(IDC_SAVE, OnSave)
        ON_WM_TIMER()
        ON_BN_CLICKED(IDC_SUSPEND, OnSuspend)
        ON_BN_CLICKED(IDC_RESUME, OnResume)
        //}}AFX_MSG_MAP
   ON_MESSAGE(WM_ANNEALINTERLEAVER_DISPLAY, OnThreadDisplay)
   ON_MESSAGE(WM_ANNEALINTERLEAVER_FINISH, OnThreadFinish)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAnnealInterleaverDlg message handlers

BOOL CAnnealInterleaverDlg::OnInitDialog()
   {
        CDialog::OnInitDialog();

        // Add "About..." menu item to system menu.

        // IDM_ABOUTBOX must be in the system command range.
        ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
        ASSERT(IDM_ABOUTBOX < 0xF000);

        CMenu* pSysMenu = GetSystemMenu(FALSE);
        if (pSysMenu != NULL)
           {
                CString strAboutMenu;
                strAboutMenu.LoadString(IDS_ABOUTBOX);
                if (!strAboutMenu.IsEmpty())
                   {
                        pSysMenu->AppendMenu(MF_SEPARATOR);
                        pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
                   }
           }

        // Set the icon for this dialog.  The framework does this automatically
        //  when the application's main window is not a dialog
        SetIcon(m_hIcon, TRUE);                 // Set big icon
        SetIcon(m_hIcon, FALSE);                // Set small icon

        // Add extra initialization here
        m_bSystemPresent = false;
   m_nSets = 1;
   m_nTau = 32;
   m_nM = 2;
   m_nType = 7;
   m_nSeed = 0;
   m_dInitTemp = 1;
   m_dFinalTemp = 1E-6;
   m_nMinIter = 100;
   m_nMinChanges = 10;
   m_dRate = 0.90;

   UpdateData(false);
   UpdateButtons(false);
   ResetDisplay();

        return TRUE;  // return TRUE  unless you set the focus to a control
   }

void CAnnealInterleaverDlg::OnSysCommand(UINT nID, LPARAM lParam)
   {
        if ((nID & 0xFFF0) == IDM_ABOUTBOX)
           {
                CAboutDlg dlgAbout;
                dlgAbout.DoModal();
           }
        else
           {
                CDialog::OnSysCommand(nID, lParam);
           }
   }

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CAnnealInterleaverDlg::OnPaint()
   {
        if (IsIconic())
           {
                CPaintDC dc(this); // device context for painting

                SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

                // Center icon in client rectangle
                int cxIcon = GetSystemMetrics(SM_CXICON);
                int cyIcon = GetSystemMetrics(SM_CYICON);
                CRect rect;
                GetClientRect(&rect);
                int x = (rect.Width() - cxIcon + 1) / 2;
                int y = (rect.Height() - cyIcon + 1) / 2;

                // Draw the icon
                dc.DrawIcon(x, y, m_hIcon);
           }
        else
           {
      // now call the base class painter
                CDialog::OnPaint();
           }
   }

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CAnnealInterleaverDlg::OnQueryDragIcon()
   {
        return (HCURSOR) m_hIcon;
   }

/////////////////////////////////////////////////////////////////////////////
// User Interface main functions

void CAnnealInterleaverDlg::OnOK()
   {
        //CDialog::OnOK();
   }

void CAnnealInterleaverDlg::OnCancel()
   {
   if(ThreadWorking())
      {
      switch(MessageBox("Worker Thread has not yet stopped - do you want to stop it and quit?", NULL, MB_YESNO | MB_ICONWARNING))
         {
         case IDYES:
            ThreadStop();
            break;
         default:
            return;
         }
      ThreadWaitFinish();
      }

   if(m_bSystemPresent)
      delete m_system;

   CDialog::OnCancel();
   }

void CAnnealInterleaverDlg::OnSave()
   {
   CString fname;
   fname.Format("sai-%d_%d-type%d_%d-%s-seed%d.txt", m_nTau, m_nM, m_nType+2, m_nSets, m_bTerm ? "term" : "noterm", m_nSeed);
   CFileDialog dlg(FALSE, "txt", fname);
   if(dlg.DoModal() == IDOK)
      {
      std::ofstream file(dlg.GetPathName());
      file << "# Interleaver Parameters:\n";
      file << "#% Sets = " << m_nSets << "\n";
      file << "#% Tau = " << m_nTau << "\n";
      file << "#% Encoder Memory = " << m_nM << "\n";
      file << "#% Type = " << m_nType+2 << "\n";
      file << "#\n";
      file << "# Process Parameters:\n";
      file << "#% Seed = " << m_nSeed << "\n";
      file << "#% Initial Temp = " << m_dInitTemp << "\n";
      file << "#% Final Temp = " << m_dFinalTemp << "\n";
      file << "#% Min Iterations = " << m_nMinIter << "\n";
      file << "#% Min Changes = " << m_nMinChanges << "\n";
      file << "#% Anneal Rate = " << m_dRate << "\n";
      file << "#\n";
      file << "#% Date: " << libbase::timer::date() << "\n";
      file << "#% Setup Time: " << m_tSetup << "\n";
      file << "#% Simulation Time: " << m_tSimulation << "\n";
      file << "#\n";
      file << *m_system;
      file.close();
      }
   }

/////////////////////////////////////////////////////////////////////////////
// User Interface message functions

void CAnnealInterleaverDlg::OnTimer(UINT_PTR nIDEvent)
   {
   CString sTemp;

   switch(nIDEvent)
      {
      case 1:
         GetDlgItem(IDC_TIME)->SetWindowText(std::string(m_tSimulation).c_str());
         //sTemp.Format("%0.1f%%", m_tSimulation.usage());
         //GetDlgItem(IDC_CPU)->SetWindowText(sTemp);
         break;
      }

        //CDialog::OnTimer(nIDEvent);
   }

LRESULT CAnnealInterleaverDlg::OnThreadDisplay(WPARAM wParam, LPARAM lParam)
   {
   CString sTemp;

   sTemp.Format("%0.5g", m_dTemp);
   GetDlgItem(IDC_TEMP)->SetWindowText(sTemp);
   sTemp.Format("%0.5g", m_dMean);
   GetDlgItem(IDC_MEAN)->SetWindowText(sTemp);
   sTemp.Format("%0.5g", m_dSigma);
   GetDlgItem(IDC_SD)->SetWindowText(sTemp);
   sTemp.Format("%0.5g", m_dHi);
   GetDlgItem(IDC_HI)->SetWindowText(sTemp);
   sTemp.Format("%0.5g", m_dLo);
   GetDlgItem(IDC_LO)->SetWindowText(sTemp);
   sTemp.Format("%0.1f%%", m_dPercent);
   GetDlgItem(IDC_PERCENT)->SetWindowText(sTemp);

   double progress = (log(m_dTemp)-log(Tstart))/(log(Tstop)-log(Tstart));
   m_pcProgress.SetPos(int(floor(100*progress)));

   libwin::CStatusGraph::Insert(GetDlgItem(IDC_GRAPH1), m_dMean);
   libwin::CStatusGraph::Insert(GetDlgItem(IDC_GRAPH2), m_dSigma);

   return 0;
   }

LRESULT CAnnealInterleaverDlg::OnThreadFinish(WPARAM wParam, LPARAM lParam)
   {
   ThreadWaitFinish();
   UpdateButtons(false);

   return 0;
   }

/////////////////////////////////////////////////////////////////////////////
// User Interface helper functions

void CAnnealInterleaverDlg::UpdateButtons(const bool bWorking)
   {
   GetDlgItem(IDC_SAVE)->EnableWindow(!bWorking & m_bSystemPresent);
   GetDlgItem(IDC_START)->EnableWindow(!bWorking);
   GetDlgItem(IDC_STOP)->EnableWindow(bWorking);
   GetDlgItem(IDC_SUSPEND)->EnableWindow(bWorking);
   GetDlgItem(IDC_RESUME)->EnableWindow(false);

   GetDlgItem(IDC_SETS)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TAU)->EnableWindow(!bWorking);
   GetDlgItem(IDC_M)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TYPE)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TERM)->EnableWindow(!bWorking);

   GetDlgItem(IDC_SEED)->EnableWindow(!bWorking);
   GetDlgItem(IDC_INITTEMP)->EnableWindow(!bWorking);
   GetDlgItem(IDC_FINALTEMP)->EnableWindow(!bWorking);
   GetDlgItem(IDC_MINITER)->EnableWindow(!bWorking);
   GetDlgItem(IDC_MINCHANGES)->EnableWindow(!bWorking);
   GetDlgItem(IDC_RATE)->EnableWindow(!bWorking);
   }

void CAnnealInterleaverDlg::ResetDisplay()
   {
   GetDlgItem(IDC_TEMP)->SetWindowText("");
   GetDlgItem(IDC_MEAN)->SetWindowText("");
   GetDlgItem(IDC_SD)->SetWindowText("");
   GetDlgItem(IDC_HI)->SetWindowText("");
   GetDlgItem(IDC_LO)->SetWindowText("");
   GetDlgItem(IDC_PERCENT)->SetWindowText("");
   GetDlgItem(IDC_TIME)->SetWindowText("");
   GetDlgItem(IDC_CPU)->SetWindowText("");
   m_pcProgress.SetPos(0);
   libwin::CStatusGraph::Reset(GetDlgItem(IDC_GRAPH1));
   libwin::CStatusGraph::Reset(GetDlgItem(IDC_GRAPH2));
   }

/////////////////////////////////////////////////////////////////////////////
// Thread process & functions called within thread process

bool CAnnealInterleaverDlg::interrupt()
   {
   return ThreadInterrupted();
   }

void CAnnealInterleaverDlg::display(const double T, const double percent, const libbase::rvstatistics E)
   {
   m_dTemp = T;
   m_dMean = E.mean();
   m_dSigma = E.sigma();
   m_dHi = E.hi();
   m_dLo = E.lo();
   m_dPercent = percent;
   SendMessage(WM_ANNEALINTERLEAVER_DISPLAY);
   }

void CAnnealInterleaverDlg::ThreadProc()
   {
   // System Setup
   m_tSetup.start();
   if(m_bSystemPresent)
      delete m_system;
   m_bSystemPresent = true;
   m_system = new libcomm::anneal_interleaver(m_nSets, m_nTau, m_nM, m_nType+2, m_bTerm!=0);
   attach_system(*m_system);
   const double E = m_system->energy();
   set_temperature(E*m_dInitTemp, E*m_dFinalTemp);
   set_iterations(m_nMinIter*m_nTau, m_nMinChanges*m_nTau);
   set_schedule(m_dRate);
   libbase::randgen prng;
   prng.seed(m_nSeed);
   seedfrom(prng);
   m_tSetup.stop();

   // Anneal Process
   m_tSimulation.start();
   SetTimer(1, 500, NULL);
   improve();
   KillTimer(1);
   m_tSimulation.stop();

   PostMessage(WM_ANNEALINTERLEAVER_FINISH);
   }

/////////////////////////////////////////////////////////////////////////////
// Thread control functions

void CAnnealInterleaverDlg::OnStart()
   {
   UpdateData(true);
   UpdateButtons(true);
   ResetDisplay();
   ThreadStart();
   }

void CAnnealInterleaverDlg::OnStop()
   {
   ThreadStop();
   }

void CAnnealInterleaverDlg::OnSuspend()
   {
   GetDlgItem(IDC_SUSPEND)->EnableWindow(false);
   GetDlgItem(IDC_RESUME)->EnableWindow(true);
        ThreadSuspend();
   }

void CAnnealInterleaverDlg::OnResume()
   {
   GetDlgItem(IDC_SUSPEND)->EnableWindow(true);
   GetDlgItem(IDC_RESUME)->EnableWindow(false);
        ThreadResume();
   }
