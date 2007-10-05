// SimulateTurboDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SimulateTurbo.h"
#include "SimulateTurboDlg.h"
#include "StatusGraph.h"

#include "awgn.h"
#include "laplacian.h"

#include "bpsk.h"
#include "qpsk.h"

#include "rscc.h"
#include "nrcc.h"

#include "puncture_null.h"
#include "puncture_stipple.h"

#include "stream_lut.h"

#include "turbo.h"
#include "logreal.h"
#include "randgen.h"
#include "commsys.h"
#include "timer.h"

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
// CSimulateTurboDlg dialog

CSimulateTurboDlg::CSimulateTurboDlg(CWnd* pParent /*=NULL*/)
: CDialog(CSimulateTurboDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CSimulateTurboDlg)
   m_nAccuracy = 0;
   m_nConfidence = 0;
   m_dSNRmax = 0.0;
   m_dSNRmin = 0.0;
   m_dSNRstep = 0.0;
   m_nIterations = 0;
   m_bParallel = FALSE;
   m_nModulation = -1;
   m_nCodeType = -1;
   m_nInputs = 0;
   m_nOutputs = 0;
   m_sGenerators = _T("");
   m_nMemory = 0;
   m_nSets = 0;
   m_bFast = FALSE;
   m_nPuncturing = -1;
   m_nTau = 0;
	m_nChannel = -1;
	m_sPathName = _T("");
	m_bEndAtZero = FALSE;
	m_bTermJPL = FALSE;
	m_bTermSimile = FALSE;
	//}}AFX_DATA_INIT
   // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
   m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
   }

void CSimulateTurboDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CSimulateTurboDlg)
   DDX_Control(pDX, IDC_PROGRESS_TOTAL, m_pcTotal);
   DDX_Control(pDX, IDC_PROGRESS_CURRENT, m_pcCurrent);
   DDX_Text(pDX, IDC_ACCURACY, m_nAccuracy);
   DDX_Text(pDX, IDC_CONFIDENCE, m_nConfidence);
   DDX_Text(pDX, IDC_SNR_MAX, m_dSNRmax);
   DDX_Text(pDX, IDC_SNR_MIN, m_dSNRmin);
   DDX_Text(pDX, IDC_SNR_STEP, m_dSNRstep);
   DDX_Text(pDX, IDC_ITERATIONS, m_nIterations);
   DDX_Check(pDX, IDC_PARALLEL, m_bParallel);
   DDX_CBIndex(pDX, IDC_MODULATION, m_nModulation);
   DDX_CBIndex(pDX, IDC_CODETYPE, m_nCodeType);
   DDX_Text(pDX, IDC_INPUTS, m_nInputs);
   DDX_Text(pDX, IDC_OUTPUTS, m_nOutputs);
   DDX_Text(pDX, IDC_GENERATORS, m_sGenerators);
   DDX_Text(pDX, IDC_MEMORY, m_nMemory);
   DDX_Text(pDX, IDC_SETS, m_nSets);
   DDX_Check(pDX, IDC_FAST, m_bFast);
   DDX_CBIndex(pDX, IDC_PUNCTURING, m_nPuncturing);
   DDX_Text(pDX, IDC_TAU, m_nTau);
	DDX_CBIndex(pDX, IDC_CHANNEL, m_nChannel);
	DDX_Text(pDX, IDC_PATHNAME, m_sPathName);
	DDX_Check(pDX, IDC_ENDATZERO, m_bEndAtZero);
	DDX_Check(pDX, IDC_TERMJPL, m_bTermJPL);
	DDX_Check(pDX, IDC_TERMSIMILE, m_bTermSimile);
	//}}AFX_DATA_MAP
   }

BEGIN_MESSAGE_MAP(CSimulateTurboDlg, CDialog)
   //{{AFX_MSG_MAP(CSimulateTurboDlg)
   ON_WM_SYSCOMMAND()
   ON_WM_PAINT()
   ON_WM_QUERYDRAGICON()
   ON_BN_CLICKED(IDC_LOAD, OnLoad)
	ON_BN_CLICKED(IDC_START, OnStart)
	ON_BN_CLICKED(IDC_STOP, OnStop)
	ON_BN_CLICKED(IDC_SAVE, OnSave)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSimulateTurboDlg message handlers

BOOL CSimulateTurboDlg::OnInitDialog()
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
   SetIcon(m_hIcon, TRUE);			// Set big icon
   SetIcon(m_hIcon, FALSE);		// Set small icon
   
   // TODO: Add extra initialization here

   // Initialise default values
   m_sResults.bPresent = false;
   UpdateButtons(false);
   ResetDisplay();

   m_nConfidence = 90;
   m_nAccuracy = 10;
   m_dSNRmin = 0.0;
   m_dSNRmax = 3.0;
   m_dSNRstep = 0.5;

   m_nChannel = 0;
   m_nModulation = 0;

   m_nIterations = 10;
   m_bParallel = false;
   m_bFast = false;

   m_nCodeType = 0;
   m_nInputs = 1;
   m_nOutputs = 2;
   m_nMemory = 2;
   m_sGenerators = "111;101";

   m_sPathName = "";
   m_nSets = 0;
   m_nTau = 0;
   m_bEndAtZero = false;
   m_bTermJPL = false;
   m_bTermSimile = false;
   m_nPuncturing = 0;

   UpdateData(false);

   // update turbo code data
   m_vpInterleavers.init(0);

   return TRUE;  // return TRUE  unless you set the focus to a control
   }

void CSimulateTurboDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CSimulateTurboDlg::OnPaint() 
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
HCURSOR CSimulateTurboDlg::OnQueryDragIcon()
   {
   return (HCURSOR) m_hIcon;
   }

void CSimulateTurboDlg::OnLoad() 
   {
   CFileDialog dlg(TRUE, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      UpdateData(true);

      // get filename and open file
      m_sPathName = dlg.GetFileName();
      FILE *file = fopen(dlg.GetPathName(), "rb");

      // get number of sets and interleaver size
      char buf[256];
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(strstr(buf,"Sets") == NULL);
      m_nSets = atoi(strchr(buf,'=')+1);
      m_bParallel = (m_nSets > 1);
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(strstr(buf,"Tau") == NULL);
      m_nTau = atoi(strchr(buf,'=')+1) + (m_bTermJPL ? m_nMemory : 0);
      
      // delete any old interleaver data
      int i;
      for(i=0; i<m_vpInterleavers.size(); i++)
         delete m_vpInterleavers(i);

      // load new interleavers
      m_vpInterleavers.init(m_nSets);
      for(i=0; i<m_vpInterleavers.size(); i++)
         m_vpInterleavers(i) = new stream_lut(m_sPathName, file, m_nTau, m_bTermJPL ? m_nMemory : 0);

      // close file
      fclose(file);
      
      // update dialog display with known items
      UpdateData(false);
      }
   }

void CSimulateTurboDlg::OnStart() 
   {
   UpdateData(true);
   ThreadStart();
   }

void CSimulateTurboDlg::OnStop() 
   {
   ThreadStop();
   }

void CSimulateTurboDlg::display(const int pass, const double cur_accuracy, const double cur_mean)
   {
   CStatusGraph::Insert(GetDlgItem(IDC_GRAPH1), cur_accuracy);
   CStatusGraph::Insert(GetDlgItem(IDC_GRAPH2), cur_mean);

   CString str;

   str.Format("%d", pass);
   GetDlgItem(IDC_DISP_PASSES)->SetWindowText(str);
   str.Format("%0.1f%%", cur_accuracy);
   GetDlgItem(IDC_DISP_ACCURACY)->SetWindowText(str);
   str.Format("%0.4g", cur_mean);
   GetDlgItem(IDC_DISP_MEAN)->SetWindowText(str);

   m_pcCurrent.SetPos(int(floor(100 - 100 * log(cur_accuracy / m_nAccuracy) / log(100 / m_nAccuracy) )));
   }

bool CSimulateTurboDlg::interrupt()
   {
   return ThreadInterrupted();
   }

void CSimulateTurboDlg::ThreadProc()
   {
   // update button status to disallow further input and reset display
   UpdateButtons(true);
   ResetDisplay();

   // System setup
   m_sResults.tSetup.start();

   // Channel Model
   channel *chan;
   switch(m_nChannel)
      {
      case 0:
         chan = new awgn;
         break;
      case 1:
         chan = new laplacian;
         break;
      }

   // Modulation Scheme
   modulator *modem;
   switch(m_nModulation)
      {
      case 0:
         modem = new bpsk;
         break;
      case 1:
         modem = new qpsk;
         break;
      }

   // Constituent Codes
   matrix<bitfield> gen(m_nInputs, m_nOutputs);
   int ndx = 0;
   for(int i=0; i<m_nInputs; i++)
      for(int j=0; j<m_nOutputs; j++)
         {
         CString thisgen = m_sGenerators.Mid(ndx).SpanExcluding(";");
         gen(i,j) = LPCSTR(thisgen);
         ndx = m_sGenerators.Find(";", ndx)+1;
         }
   fsm *encoder;
   switch(m_nCodeType)
      {
      case 0:
         encoder = new rscc(m_nInputs, m_nOutputs, gen);
         break;
      case 1:
         encoder = new nrcc(m_nInputs, m_nOutputs, gen);
         break;
      }

   // Puncturing system
   puncture *punc;
   switch(m_nPuncturing)
      {
      case 0:
         punc = new puncture_null(m_nTau, 1+(m_nSets+1)*(m_nOutputs-m_nInputs));
         break;
      case 1:
         punc = new puncture_stipple(m_nTau, 1+(m_nSets+1)*(m_nOutputs-m_nInputs));
         break;
      }

   // Channel Codec (punctured, iterations, simile, endatzero)
   turbo<logreal> codec(*encoder, *modem, *punc, *chan, m_nTau, m_vpInterleavers, m_nIterations, m_bTermSimile!=0, m_bEndAtZero!=0, m_bParallel!=0);

   // Store information on system to be simulated
   m_sResults.nConfidence = m_nConfidence;
   m_sResults.nAccuracy = m_nAccuracy;
   m_sResults.nChannel = m_nChannel;
   m_sResults.nModulation = m_nModulation;
   m_sResults.nIterations = m_nIterations;
   m_sResults.bFast = m_bFast!=0;
   CStringStreamBuf sbCodec(&m_sResults.sCodec);
   ostream osCodec(&sbCodec);
   osCodec << codec;
   m_sResults.sFilename.Format("turbo-%d_%d-%s-%s", codec.block_outbits(), codec.block_inbits(), m_sGenerators, m_sPathName);
   m_sResults.sFilename.Replace(';','_');

   // Source Generator
   randgen src;
   // The complete communication system
   commsys system(&src, chan, &codec, m_bFast!=0);

   m_sResults.tSetup.stop();

   // The actual estimator - tie to the system and set up
   initialise(&system);
   set_confidence(m_nConfidence/double(100));
   set_accuracy(m_nAccuracy/double(100));

   // Initialise Results tables
   const int iSteps = int(floor((m_dSNRmax-m_dSNRmin)/m_dSNRstep))+1;
   m_sResults.vdSNR.init(iSteps);
   m_sResults.viSamples.init(iSteps);
   m_sResults.mdEstimate.init(iSteps, system.count());
   m_sResults.mdError.init(iSteps, system.count());

   // Work out the following for every SNR value required
   m_sResults.tSimulation.start();
   m_pcTotal.SetRange(0, iSteps);
   int iStep = 0;
   for(double SNR=m_dSNRmin; SNR<=m_dSNRmax; SNR+=m_dSNRstep)
      {
      chan->set_snr(SNR);

      CStatusGraph::Reset(GetDlgItem(IDC_GRAPH1));
      CStatusGraph::Reset(GetDlgItem(IDC_GRAPH2));

      clog << "Simulating system at Eb/No = " << SNR << "\n" << flush;
      vector<double> est, tol;
      estimate(est, tol);
      
      m_sResults.vdSNR(iStep) = SNR;
      m_sResults.viSamples(iStep) = get_samplecount();
      for(i=0; i<system.count(); i++)
         {
         m_sResults.mdEstimate(iStep, i) = est(i);
         m_sResults.mdError(iStep, i) = est(i)*tol(i);
         }

      m_pcTotal.SetPos(++iStep);
      }
   m_sResults.tSimulation.stop();

   // release the estimator so we can use it again
   finalise();

   // clean up allocated objects
   delete punc;
   delete encoder;
   delete modem;
   delete chan;

   // indicate that we have results to save and update button status
   m_sResults.bPresent = true;
   UpdateButtons(false);
   }

void CSimulateTurboDlg::OnSave() 
   {
   CFileDialog dlg(FALSE, "txt", m_sResults.sFilename, OFN_HIDEREADONLY);
   if(dlg.DoModal() == IDOK)
      {
      ofstream file(dlg.GetPathName(), ios::out | ios::app);

      // Print information on the statistical accuracy of results being worked
      if(ThreadInterrupted())
         file << "## *** NOTE: SIMULATION INTERRUPTED - RESULTS MAY BE INACCURATE ***\n";
      file << "#% Date: " << timer::date() << "\n";
      file << "#% Setup Time: " << m_sResults.tSetup << "\n";
      file << "#% Simulation Time: " << m_sResults.tSimulation << "\n";
      file << "#\n";
      file << "#% Tolerance: " << m_sResults.nAccuracy << "%\n";
      file << "#% Confidence: " << m_sResults.nConfidence << "%\n";
      file << "#% Iterations: " << m_sResults.nIterations << (m_sResults.bFast ? " (Fast Cutoff)" : "") << "\n";
      file << "#% Channel: " << (m_sResults.nChannel==0 ? "AWGN" : "Laplacian") << "\n";
      file << "#% Modulation: " << (m_sResults.nModulation==0 ? "BPSK" : "QPSK") << "\n";
      file << "#% Codec: " << m_sResults.sCodec << "\n";
      file << "#\n";
      
      // Print the results
      file.precision(6);
      const int iSteps = m_sResults.vdSNR.size();
      for(int iStep=0; iStep<iSteps; iStep++)
         {
         file << m_sResults.vdSNR(iStep) << "\t";
         for(int i=0; i<m_sResults.mdEstimate.ysize(); i++)
            {
            file << m_sResults.mdEstimate(iStep, i) << "\t";
            file << m_sResults.mdError(iStep, i) << "\t";
            }
         file << m_sResults.viSamples(iStep) << "\n";
         }

      file.close();
      }
   }

void CSimulateTurboDlg::ResetDisplay()
   {
   GetDlgItem(IDC_DISP_PASSES)->SetWindowText("");
   GetDlgItem(IDC_DISP_ACCURACY)->SetWindowText("");
   GetDlgItem(IDC_DISP_MEAN)->SetWindowText("");
   m_pcCurrent.SetPos(0);
   m_pcTotal.SetPos(0);
   CStatusGraph::Reset(GetDlgItem(IDC_GRAPH1));
   CStatusGraph::Reset(GetDlgItem(IDC_GRAPH2));
   }

void CSimulateTurboDlg::UpdateButtons(const bool bWorking)
   {
   GetDlgItem(IDC_START)->EnableWindow(!bWorking);
   GetDlgItem(IDC_STOP)->EnableWindow(bWorking);
   GetDlgItem(IDC_SAVE)->EnableWindow(!bWorking & m_sResults.bPresent);

   GetDlgItem(IDC_CONFIDENCE)->EnableWindow(!bWorking);
   GetDlgItem(IDC_ACCURACY)->EnableWindow(!bWorking);
   GetDlgItem(IDC_SNR_MIN)->EnableWindow(!bWorking);
   GetDlgItem(IDC_SNR_MAX)->EnableWindow(!bWorking);
   GetDlgItem(IDC_SNR_STEP)->EnableWindow(!bWorking);

   GetDlgItem(IDC_CHANNEL)->EnableWindow(!bWorking);
   GetDlgItem(IDC_MODULATION)->EnableWindow(!bWorking);

   GetDlgItem(IDC_ITERATIONS)->EnableWindow(!bWorking);
   GetDlgItem(IDC_PARALLEL)->EnableWindow(!bWorking);
   GetDlgItem(IDC_FAST)->EnableWindow(!bWorking);

   GetDlgItem(IDC_CODETYPE)->EnableWindow(!bWorking);
   GetDlgItem(IDC_INPUTS)->EnableWindow(!bWorking);
   GetDlgItem(IDC_OUTPUTS)->EnableWindow(!bWorking);
   GetDlgItem(IDC_MEMORY)->EnableWindow(!bWorking);
   GetDlgItem(IDC_GENERATORS)->EnableWindow(!bWorking);

   GetDlgItem(IDC_PATHNAME)->EnableWindow(!bWorking);
   GetDlgItem(IDC_LOAD)->EnableWindow(!bWorking);
   GetDlgItem(IDC_SETS)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TAU)->EnableWindow(!bWorking);
   GetDlgItem(IDC_ENDATZERO)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TERMSIMILE)->EnableWindow(!bWorking);
   GetDlgItem(IDC_TERMJPL)->EnableWindow(!bWorking);
   GetDlgItem(IDC_PUNCTURING)->EnableWindow(!bWorking);
   }

void CSimulateTurboDlg::OnOK() 
   {
   //CDialog::OnOK();
   }

void CSimulateTurboDlg::OnCancel() 
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
