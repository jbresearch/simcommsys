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

#include "stdafx.h"
#include "PluginWizard.h"
#include "PluginWizardDlg.h"
#include "vector.h"
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
// CPluginWizardDlg dialog

CPluginWizardDlg::CPluginWizardDlg(CWnd* pParent /*=NULL*/)
: CDialog(CPluginWizardDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CPluginWizardDlg)
   m_nType = 0;
        m_sNewName = _T("");
        m_sOldName = _T("");
        //}}AFX_DATA_INIT
   // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
   m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
   }

void CPluginWizardDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CPluginWizardDlg)
   DDX_CBIndex(pDX, IDC_TYPE, m_nType);
        DDX_Text(pDX, IDC_NEWNAME, m_sNewName);
        DDX_Text(pDX, IDC_OLDNAME, m_sOldName);
        //}}AFX_DATA_MAP
   }

BEGIN_MESSAGE_MAP(CPluginWizardDlg, CDialog)
//{{AFX_MSG_MAP(CPluginWizardDlg)
ON_WM_SYSCOMMAND()
ON_WM_PAINT()
ON_WM_QUERYDRAGICON()
        ON_CBN_SELCHANGE(IDC_TYPE, OnSelchangeType)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CPluginWizardDlg message handlers

BOOL CPluginWizardDlg::OnInitDialog()
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

   // Extra initialization
   GetDlgItem(IDC_OLDNAME)->EnableWindow(m_nType >= 2);

   return TRUE;  // return TRUE  unless you set the focus to a control
   }

void CPluginWizardDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CPluginWizardDlg::OnPaint()
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
HCURSOR CPluginWizardDlg::OnQueryDragIcon()
   {
   return (HCURSOR) m_hIcon;
   }

void CPluginWizardDlg::OnOK()
   {
   UpdateData(true);

   // Create input file lists and replaceable names
   switch(m_nType)
      {
      case 0:
         m_sOldName = "Shell";
         break;
      case 1:
         m_sOldName = "MFCShell";
         break;
      case 2:
         break;
      default:
         AfxMessageBox("Type chosen not yet implemented");
         return;
      }
   UpdateData(false);

   // Validate name and create directory for target project
   if(m_sNewName.IsEmpty() || m_sNewName.FindOneOf(" ")>=0)
      {
      AfxMessageBox("Please choose a valid project name");
      return;
      }
   if(CreateDirectory(m_sPath+m_sNewName, NULL) == 0)
      {
      AfxMessageBox("Directory creation failed on "+m_sPath+m_sNewName+" - choose another project name");
      return;
      }

   // Now process each file in turn
   CFileFind finder;
   BOOL bWorking = finder.FindFile(m_sPath+m_sOldName+"\\*.*");
   while(bWorking)
      {
      bWorking = finder.FindNextFile();
      // skip parent & current dir entries
      if(finder.IsDots())
         continue;
      // skip if directory (we do not want to recurse)
      if(finder.IsDirectory())
         continue;
      // check if it's the correct extension
      const CString sType = "*.h;*.cpp;*.dsp;*.r;*.rc";
      const CString sExtension = finder.GetFileName().Right(finder.GetFileName().GetLength() - finder.GetFileTitle().GetLength());
      if(sType.Find(sExtension) < 0)
         continue;

      // determine input and output filenames
      CString sFileName = finder.GetFileName();
      std::ifstream fileIn(m_sPath+m_sOldName+"\\"+sFileName);
      sFileName.Replace(m_sOldName, m_sNewName);
      std::ofstream fileOut(m_sPath+m_sNewName+"\\"+sFileName);

      // process file
      const int buflen = 256;
      char buf[buflen];
      for(fileIn.getline(buf, buflen); !fileIn.eof(); fileIn.getline(buf, buflen))
         {
         CString str = buf;
         str.Replace(m_sOldName, m_sNewName);
         fileOut << LPCTSTR(str) << "\n";
         }
      }

   CDialog::OnOK();
   }

void CPluginWizardDlg::OnSelchangeType()
   {
   UpdateData(true);
   GetDlgItem(IDC_OLDNAME)->EnableWindow(m_nType >= 2);
   }
