// SelectCodecDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "SelectCodecDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectCodecDlg dialog


CSelectCodecDlg::CSelectCodecDlg(CWnd* pParent /*=NULL*/)
: CDialog(CSelectCodecDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CSelectCodecDlg)
   m_nMath = 0;
   m_nType = 0;
   //}}AFX_DATA_INIT
   }


void CSelectCodecDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CSelectCodecDlg)
   DDX_CBIndex(pDX, IDC_MATH, m_nMath);
   DDX_CBIndex(pDX, IDC_TYPE, m_nType);
   //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CSelectCodecDlg, CDialog)
//{{AFX_MSG_MAP(CSelectCodecDlg)
ON_CBN_SELCHANGE(IDC_TYPE, OnSelchangeType)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectCodecDlg message handlers

BOOL CSelectCodecDlg::OnInitDialog() 
   {
   CDialog::OnInitDialog();
   
   GetDlgItem(IDC_MATH)->EnableWindow(m_nType > 0);
   
   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CSelectCodecDlg::OnSelchangeType() 
   {
   UpdateData(true);
   GetDlgItem(IDC_MATH)->EnableWindow(m_nType > 0);
   }

void CSelectCodecDlg::OnOK() 
   {
   UpdateData(true);
   if(m_nType < 0)
      MessageBox("Invalid codec type!", NULL, MB_ICONWARNING | MB_OK);
        else if(m_nType > 0 && m_nMath < 0)
      MessageBox("Invalid arithmetic type!", NULL, MB_ICONWARNING | MB_OK);
   else
           CDialog::OnOK();
   }
