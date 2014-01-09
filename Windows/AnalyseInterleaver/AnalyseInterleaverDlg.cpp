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

// AnalyseInterleaverDlg.cpp : implementation file
//

#include "stdafx.h"
#include "AnalyseInterleaver.h"
#include "AnalyseInterleaverDlg.h"

#include "Histogram2D.h"
#include <math.h>

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
// CAnalyseInterleaverDlg dialog

CAnalyseInterleaverDlg::CAnalyseInterleaverDlg(CWnd* pParent /*=NULL*/)
: CDialog(CAnalyseInterleaverDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CAnalyseInterleaverDlg)
   m_sPathName = _T("");
   m_nTau = 0;
   m_nSpread = 0;
        m_nMaxDist = 0;
        //}}AFX_DATA_INIT
   // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
   m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
   }

void CAnalyseInterleaverDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CAnalyseInterleaverDlg)
   DDX_Control(pDX, IDC_PROGRESS, m_pcProgress);
   DDX_Text(pDX, IDC_PATHNAME, m_sPathName);
   DDX_Text(pDX, IDC_TAU, m_nTau);
   DDX_Text(pDX, IDC_SPREAD, m_nSpread);
        DDX_Text(pDX, IDC_MAXDIST, m_nMaxDist);
        //}}AFX_DATA_MAP
   }

BEGIN_MESSAGE_MAP(CAnalyseInterleaverDlg, CDialog)
//{{AFX_MSG_MAP(CAnalyseInterleaverDlg)
ON_WM_SYSCOMMAND()
ON_WM_PAINT()
ON_WM_QUERYDRAGICON()
ON_BN_CLICKED(IDC_LOAD, OnLoad)
ON_BN_CLICKED(IDC_ANALYSE, OnAnalyse)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CAnalyseInterleaverDlg message handlers

BOOL CAnalyseInterleaverDlg::OnInitDialog()
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

   // Add extra initialization here
   m_pInterleaver = NULL;
   m_nMaxDist = 100;
   UpdateData(false);

   return TRUE;  // return TRUE  unless you set the focus to a control
   }

void CAnalyseInterleaverDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CAnalyseInterleaverDlg::OnPaint()
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
HCURSOR CAnalyseInterleaverDlg::OnQueryDragIcon()
   {
   return (HCURSOR) m_hIcon;
   }

void CAnalyseInterleaverDlg::OnLoad()
   {
   CFileDialog dlg(TRUE, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      m_sPathName = dlg.GetFileName();
      FILE *file = fopen(dlg.GetPathName(), "rb");
      char buf[256];
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(strstr(buf,"Tau") == NULL);
      m_nTau = atoi(strchr(buf,'=')+1);

      UpdateData(false);

      if(m_pInterleaver != NULL)
         delete m_pInterleaver;
      m_pInterleaver = new libcomm::stream_lut<double>(m_sPathName, file, m_nTau, 0);
      fclose(file);
      }
   }

void CAnalyseInterleaverDlg::OnAnalyse()
   {
   GetDlgItem(IDC_ANALYSE)->EnableWindow(false);
   GetDlgItem(IDC_MAXDIST)->EnableWindow(false);

   UpdateData(true);

   int i,j;
   m_miIOSS.init(m_nMaxDist,m_nMaxDist);
   m_miIOSS = 0;

   // Generate LUT from the interleaver
   libbase::vector<int> in(m_nTau);
   for(i=0; i<m_nTau; i++)
      in(i) = i;
   libbase::vector<int> out;
   m_pInterleaver->transform(in, out);

   // Construct IOSS from LUT
   m_nSpread = m_nMaxDist;
   for(i=0; i<m_nTau; i++)
      {
      for(j=i+1; j<std::min(m_nTau,i+m_nMaxDist); j++)
         {
         const int din = (j-i);
         const int dout = int(fabs(double(out(j)-out(i))));
         if(dout < m_nMaxDist)
            m_miIOSS(din,dout)++;
         if(din <= m_nSpread && dout <= m_nSpread)
            m_nSpread = std::max(din,dout);
         }
      m_pcProgress.SetPos(int(floor(100*i/double(m_nTau))));
      }
   UpdateData(false);

   // Update display
   libwin::CHistogram2D::UpdateData(GetDlgItem(IDC_HISTOGRAM), m_miIOSS);

   GetDlgItem(IDC_ANALYSE)->EnableWindow(true);
   GetDlgItem(IDC_MAXDIST)->EnableWindow(true);
   }

void CAnalyseInterleaverDlg::OnOK()
   {
   //CDialog::OnOK();
   }

void CAnalyseInterleaverDlg::OnCancel()
   {
   CDialog::OnCancel();
   }
