#include "stdafx.h"
#include "FilterOrphans.h"
#include "FilterOrphansDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

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
// CFilterOrphansDlg dialog

CFilterOrphansDlg::CFilterOrphansDlg(CWnd* pParent /*=NULL*/)
: CDialog(CFilterOrphansDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CFilterOrphansDlg)
	m_bKeepNoise = FALSE;
	m_nWeight = 0;
	//}}AFX_DATA_INIT
   }


void CFilterOrphansDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CFilterOrphansDlg)
	DDX_Control(pDX, IDC_SLIDER, m_scSlider);
	DDX_Check(pDX, IDC_KEEPNOISE, m_bKeepNoise);
	DDX_Text(pDX, IDC_WEIGHT, m_nWeight);
	DDV_MinMaxInt(pDX, m_nWeight, 0, 8);
	//}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CFilterOrphansDlg, CDialog)
//{{AFX_MSG_MAP(CFilterOrphansDlg)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER, OnCustomdrawSlider)
	ON_EN_CHANGE(IDC_WEIGHT, OnChangeWeight)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterOrphansDlg message handlers

BOOL CFilterOrphansDlg::OnInitDialog() 
   {
   CDialog::OnInitDialog();
   
   // set slider limits & position (cf OnChangeWeight)
   m_scSlider.SetRange(0, 8);
   m_scSlider.SetPos(m_nWeight);

   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CFilterOrphansDlg::OnChangeWeight() 
   {
   m_nWeight = GetDlgItemInt(IDC_WEIGHT);
   if(m_scSlider.GetPos() != m_nWeight)
      {
      m_scSlider.SetPos(m_nWeight);
      m_scSlider.RedrawWindow();
      }
   }

void CFilterOrphansDlg::OnCustomdrawSlider(NMHDR* pNMHDR, LRESULT* pResult) 
   {
   if(m_scSlider.GetPos() != m_nWeight && m_scSlider.GetRangeMax() >= m_nWeight)
      SetDlgItemInt(IDC_WEIGHT, m_scSlider.GetPos());
   *pResult = 0;
   }
