// TestBenchDlg.cpp : implementation file
//

#include "stdafx.h"
#include "TestBench.h"
#include "TestBenchDlg.h"

#include <fstream>
#include <map>
#include <string>
using namespace std;

#include "berrou.h"
#include "flat.h"
#include "helical.h"
#include "rand_lut.h"
#include "rectangular.h"
#include "shift_lut.h"
#include "uniform_lut.h"
#include "vale96int.h"
#include "onetimepad.h"
#include "padded.h"
#include "rscc.h"

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
// CTestBenchDlg dialog

CTestBenchDlg::CTestBenchDlg(CWnd* pParent /*=NULL*/)
: CDialog(CTestBenchDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CTestBenchDlg)
	m_nType = -1;
	//}}AFX_DATA_INIT
   // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
   m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
   }

CTestBenchDlg::~CTestBenchDlg()
   {
   if(pInter != NULL)
      delete pInter;
   }

void CTestBenchDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CTestBenchDlg)
	DDX_CBIndex(pDX, IDC_TYPE, m_nType);
	//}}AFX_DATA_MAP
   }

BEGIN_MESSAGE_MAP(CTestBenchDlg, CDialog)
//{{AFX_MSG_MAP(CTestBenchDlg)
ON_WM_SYSCOMMAND()
ON_WM_PAINT()
ON_WM_QUERYDRAGICON()
ON_BN_CLICKED(IDC_LOAD, OnLoad)
ON_BN_CLICKED(IDC_SAVE, OnSave)
	ON_BN_CLICKED(IDC_MAKE, OnMake)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CTestBenchDlg message handlers

BOOL CTestBenchDlg::OnInitDialog()
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
   pInter = NULL;
   
   return TRUE;  // return TRUE  unless you set the focus to a control
   }

void CTestBenchDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CTestBenchDlg::OnPaint() 
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
HCURSOR CTestBenchDlg::OnQueryDragIcon()
   {
   return (HCURSOR) m_hIcon;
   }

void CTestBenchDlg::OnLoad() 
   {
   CFileDialog dlg(TRUE, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      if(pInter != NULL)
         delete pInter;
      ifstream file(dlg.GetPathName());
      file >> pInter;
      TRACE("Interleaver Loaded\n");
      TRACE("  pointer = 0x%x\n", pInter);
      TRACE("  name = \"%s\"\n", pInter->name());
      }
   }

void CTestBenchDlg::OnSave() 
   {
   CFileDialog dlg(FALSE, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      ofstream file(dlg.GetPathName());
      file << pInter;
      }
   }

void CTestBenchDlg::OnMake() 
   {
   UpdateData(true);
   if(pInter != NULL)
      delete pInter;

   const int k=1, n=2;
   matrix<bitfield> gen(k, n);
   gen(0, 0) = "111";
   gen(0, 1) = "101";
   const rscc encoder(gen);

   switch(m_nType)
      {
      case 0:
         pInter = new berrou(16);
         break;
      case 1:
	      pInter = new flat(16);
         break;
      case 2:
         pInter = new helical(37,5,7);
         break;
      case 3:
         pInter = new rand_lut(18,2);
         break;
      case 4:
         pInter = new rectangular(37,5,7);
         break;
      case 5:
         pInter = new shift_lut(2,16);
         break;
      case 6:
         pInter = new uniform_lut(16,2);
         break;
      case 7:
         pInter = new vale96int();
         break;
      case 8:
         pInter = new onetimepad(encoder,16,true,false);
         break;
      case 9:
         pInter = new padded(rand_lut(18,2),encoder,16,true,false);
         break;
      default:
         MessageBox("Unknown interleaver type", NULL, MB_ICONWARNING|MB_OK);
         break;
      }
   }
