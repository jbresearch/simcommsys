// SelectAccuracyDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SimulateCommsys.h"
#include "SelectAccuracyDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectAccuracyDlg dialog


CSelectAccuracyDlg::CSelectAccuracyDlg(CWnd* pParent /*=NULL*/)
: CDialog(CSelectAccuracyDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CSelectAccuracyDlg)
   m_dAccuracy = 0.0;
   m_dConfidence = 0.0;
   //}}AFX_DATA_INIT
   }


void CSelectAccuracyDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CSelectAccuracyDlg)
   DDX_Text(pDX, IDC_ACCURACY, m_dAccuracy);
   DDX_Text(pDX, IDC_CONFIDENCE, m_dConfidence);
   //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CSelectAccuracyDlg, CDialog)
//{{AFX_MSG_MAP(CSelectAccuracyDlg)
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectAccuracyDlg message handlers

void CSelectAccuracyDlg::OnOK()
   {
   // TODO: Add extra validation here
   if(!(m_dAccuracy > 0.0 && m_dAccuracy < 1.0))
      {
      MessageBox("Invalid accuracy value (must be between 0.0 and 1.0).", NULL, MB_OK | MB_ICONWARNING);
      return;
      }
   if(!(m_dConfidence > 0.0 && m_dConfidence < 1.0))
      {
      MessageBox("Invalid confidence value (must be between 0.0 and 1.0).", NULL, MB_OK | MB_ICONWARNING);
      return;
      }

   CDialog::OnOK();
   }
