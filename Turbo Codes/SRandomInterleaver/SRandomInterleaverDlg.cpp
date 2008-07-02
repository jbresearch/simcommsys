// SRandomInterleaverDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SRandomInterleaver.h"
#include "SRandomInterleaverDlg.h"
#include "randgen.h"
#include <math.h>
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


template <class T>
class DerivedVector : public libbase::vector<T>
{
public:
   DerivedVector(const int x=0);
   void remove(const int x);
   void sequence();
};

template <class T>
DerivedVector<T>::DerivedVector(const int x) : vector<T>(x)
   {
   }

template <class T>
void DerivedVector<T>::remove(const int x)
   {
   ASSERT(x < m_xsize);
   for(int i=x; i<m_xsize-1; i++)
      m_data[i] = m_data[i+1];
   m_xsize--;
   }

template <class T>
void DerivedVector<T>::sequence()
   {
   for(int i=0; i<m_xsize; i++)
      m_data[i] = i;
   }

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
// CSRandomInterleaverDlg dialog

CSRandomInterleaverDlg::CSRandomInterleaverDlg(CWnd* pParent /*=NULL*/)
: CDialog(CSRandomInterleaverDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CSRandomInterleaverDlg)
   m_nSpread = 0;
   m_nTau = 0;
   m_nSeed = 0;
   m_nAttempts = 0;
   m_nUsedSeed = 0;
   //}}AFX_DATA_INIT
   // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
   m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
   }

CSRandomInterleaverDlg::~CSRandomInterleaverDlg()
   {
   }

void CSRandomInterleaverDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CSRandomInterleaverDlg)
   DDX_Control(pDX, IDC_PROGRESS_ATTEMPT, m_pcAttempt);
   DDX_Control(pDX, IDC_PROGRESS, m_pcProgress);
   DDX_Text(pDX, IDC_SPREAD, m_nSpread);
   DDX_Text(pDX, IDC_TAU, m_nTau);
   DDX_Text(pDX, IDC_SEED, m_nSeed);
   DDX_Text(pDX, IDC_ATTEMPTS, m_nAttempts);
   DDX_Text(pDX, IDC_USED_SEED, m_nUsedSeed);
   //}}AFX_DATA_MAP
   }

BEGIN_MESSAGE_MAP(CSRandomInterleaverDlg, CDialog)
//{{AFX_MSG_MAP(CSRandomInterleaverDlg)
ON_WM_SYSCOMMAND()
ON_WM_PAINT()
ON_WM_QUERYDRAGICON()
ON_BN_CLICKED(IDC_SUGGEST, OnSuggest)
ON_BN_CLICKED(IDC_SAVE, OnSave)
ON_BN_CLICKED(IDC_START, OnStart)
ON_BN_CLICKED(IDC_STOP, OnStop)
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSRandomInterleaverDlg message handlers

BOOL CSRandomInterleaverDlg::OnInitDialog()
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
   SetIcon(m_hIcon, TRUE);                      // Set big icon
   SetIcon(m_hIcon, FALSE);             // Set small icon

   // TODO: Add extra initialization here
   m_nSeed = 0;
   m_nTau = 1024;
   m_nSpread = int(floor(sqrt(m_nTau/double(2))));
   m_nAttempts = 5;
   UpdateData(false);
   m_bValidResults = false;
   UpdateButtons(false);

   return TRUE;  // return TRUE  unless you set the focus to a control
   }

void CSRandomInterleaverDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CSRandomInterleaverDlg::OnPaint()
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
      CDialog::OnPaint();
      }
   }

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CSRandomInterleaverDlg::OnQueryDragIcon()
   {
   return (HCURSOR) m_hIcon;
   }

void CSRandomInterleaverDlg::OnStart()
   {
   UpdateData(true);
   ThreadStart();
   }

void CSRandomInterleaverDlg::OnStop()
   {
   ThreadStop();
   }

void CSRandomInterleaverDlg::OnSuggest()
   {
   UpdateData(true);
   m_nSpread = int(floor(sqrt(m_nTau/double(2))));
   UpdateData(false);
   }

void CSRandomInterleaverDlg::ThreadProc()
   {
   UpdateButtons(true);

   m_bValidResults = false;
   m_tDuration.start();
   m_viInterleaver.init(m_nTau);
   m_pcProgress.SetRange32(0, m_nTau);
   m_pcAttempt.SetRange32(0, m_nAttempts);

   bool failed;
   int attempt = 0;
   // loop for a number of attempts at the given Spread, then
   // reduce and continue as necessary
   do {
      // set up for the current attempt
      m_pcAttempt.SetPos(attempt);
      m_nUsedSeed = m_nSeed+attempt;
      UpdateData(false);
      libbase::randgen src;
      src.seed(m_nUsedSeed);
      DerivedVector<int> unused(m_nTau);
      unused.sequence();
      failed = false;
      // loop to fill all entries in the interleaver - or until we fail
      libbase::timer t;
      for(int i=0; i<m_nTau && !failed; i++)
         {
         // handle interruptions here
         if(ThreadInterrupted())
            {
            UpdateButtons(false);
            return;
            }
         // keep user happy
         if(t.elapsed() > 0.1)
            {
            m_pcProgress.SetPos(i);
            t.start();
            }
         // set up for the current entry
         DerivedVector<int> untried = unused;
         DerivedVector<int> index(unused.size());
         index.sequence();
         int n, ndx;
         bool good;
         // loop for the current entry - until we manage to find a suitable value
         // or totally fail in trying
         do {
            // choose a random number from what's left to try
            ndx = src.ival(untried.size());
            n = untried(ndx);
            // see if it's a suitable value (ie satisfies spread constraint)
            good = true;
            for(int j=max(0,i-m_nSpread); j<i; j++)
               if(abs(m_viInterleaver(j)-n) < m_nSpread)
                  {
                  good = false;
                  break;
                  }
            // if it's no good remove it from the list of options,
            // if it's good then insert it into the interleaver & mark that number as used
            if(!good)
               {
               untried.remove(ndx);
               index.remove(ndx);
               failed = (untried.size() == 0);
               }
            else
               {
               unused.remove(index(ndx));
               m_viInterleaver(i) = n;
               }
            } while(!good && !failed);
         }
      // if this failed, prepare for the next attempt
      if(failed)
         {
         attempt++;
         if(attempt >= m_nAttempts)
            {
            attempt = 0;
            m_nSpread--;
            UpdateData(false);
            }
         }
      } while(failed);

   m_tDuration.stop();
   m_bValidResults = true;
   m_nUsedSpread = m_nSpread;

   UpdateButtons(false);
   }

void CSRandomInterleaverDlg::UpdateButtons(const bool bWorking)
   {
   GetDlgItem(IDC_SAVE)->EnableWindow(!bWorking && m_bValidResults);
   GetDlgItem(IDC_SUGGEST)->EnableWindow(!bWorking);
   GetDlgItem(IDC_START)->EnableWindow(!bWorking);
   GetDlgItem(IDC_STOP)->EnableWindow(bWorking);

   GetDlgItem(IDC_SEED)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TAU)->EnableWindow(!bWorking);
   GetDlgItem(IDC_SPREAD)->EnableWindow(!bWorking);
   GetDlgItem(IDC_ATTEMPTS)->EnableWindow(!bWorking);
   }

void CSRandomInterleaverDlg::OnSave()
   {
   CString fname;
   fname.Format("sri-%d-spread%d-seed%d.txt", m_nTau, m_nSpread, m_nUsedSeed);
   CFileDialog dlg(FALSE, "txt", fname);
   if(dlg.DoModal() == IDOK)
      {
      std::ofstream file(dlg.GetPathName());
      file << "# Interleaver Parameters:\n";
      file << "#% Sets = 1\n";
      file << "#% Tau = " << m_viInterleaver.size() << "\n";
      file << "#% Spread = " << m_nUsedSpread << "\n";
      file << "#\n";
      file << "# Process Parameters:\n";
      file << "#% Seed = " << m_nUsedSeed << "\n";
      file << "#\n";
      file << "#% Date: " << libbase::timer::date() << "\n";
      file << "#% Time Taken: " << m_tDuration << "\n";
      file << "#\n";
      for(int i=0; i<m_viInterleaver.size(); i++)
         file << m_viInterleaver(i) << "\n";
      file.close();
      }
   }

void CSRandomInterleaverDlg::OnOK()
   {
   //CDialog::OnOK();
   }

void CSRandomInterleaverDlg::OnCancel()
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

   CDialog::OnCancel();
   }
