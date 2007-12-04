// SelectRangeDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SimulateCommsys.h"
#include "SelectRangeDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectRangeDlg dialog


CSelectRangeDlg::CSelectRangeDlg(CWnd* pParent /*=NULL*/)
: CDialog(CSelectRangeDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CSelectRangeDlg)
   m_dSNRmax = 0.0;
   m_dSNRmin = 0.0;
   m_dSNRstep = 0.0;
   //}}AFX_DATA_INIT
   }


void CSelectRangeDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CSelectRangeDlg)
   DDX_Text(pDX, IDC_SNR_MAX, m_dSNRmax);
   DDX_Text(pDX, IDC_SNR_MIN, m_dSNRmin);
   DDX_Text(pDX, IDC_SNR_STEP, m_dSNRstep);
   //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CSelectRangeDlg, CDialog)
//{{AFX_MSG_MAP(CSelectRangeDlg)
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectRangeDlg message handlers

void CSelectRangeDlg::OnOK() 
   {
   // TODO: Add extra validation here
   if(m_dSNRmin > m_dSNRmax)
      {
      MessageBox("SNR max must be greater or equal to SNR min.", NULL, MB_OK | MB_ICONWARNING);
      return;
      }
   if(m_dSNRstep <= 0)
      {
      MessageBox("SNR step must be a positive value.", NULL, MB_OK | MB_ICONWARNING);
      return;
      }
   
   CDialog::OnOK();
   }
