// SelectGeneratorDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "SelectGeneratorDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectGeneratorDlg dialog


CSelectGeneratorDlg::CSelectGeneratorDlg(CWnd* pParent /*=NULL*/)
: CDialog(CSelectGeneratorDlg::IDD, pParent)
   {
   //{{AFX_DATA_INIT(CSelectGeneratorDlg)
   m_sValue = _T("");
   //}}AFX_DATA_INIT
   }


void CSelectGeneratorDlg::DoDataExchange(CDataExchange* pDX)
   {
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CSelectGeneratorDlg)
   DDX_Control(pDX, IDC_OUTPUT, m_scOutput);
   DDX_Control(pDX, IDC_INPUT, m_scInput);
   DDX_Text(pDX, IDC_VALUE, m_sValue);
   //}}AFX_DATA_MAP
   }


BEGIN_MESSAGE_MAP(CSelectGeneratorDlg, CDialog)
//{{AFX_MSG_MAP(CSelectGeneratorDlg)
        ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_INPUT, OnReleasedcaptureInput)
        ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_OUTPUT, OnReleasedcaptureOutput)
        ON_EN_UPDATE(IDC_VALUE, OnUpdateValue)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectGeneratorDlg message handlers

BOOL CSelectGeneratorDlg::OnInitDialog() 
   {
   CDialog::OnInitDialog();
   
   m_scInput.SetRange(0, m_mbGenerator.xsize()-1);
   m_scOutput.SetRange(0, m_mbGenerator.ysize()-1);
   m_scInput.SetPos(0);
   m_scOutput.SetPos(0);
   UpdateValue();
   
   return TRUE;  // return TRUE unless you set the focus to a control
   // EXCEPTION: OCX Property Pages should return FALSE
   }

void CSelectGeneratorDlg::UpdateValue()
   {
   m_sValue = std::string(m_mbGenerator(m_scInput.GetPos(), m_scOutput.GetPos())).c_str();
   UpdateData(false);
   }

void CSelectGeneratorDlg::OnReleasedcaptureInput(NMHDR* pNMHDR, LRESULT* pResult) 
   {
   UpdateValue();
   *pResult = 0;
   }

void CSelectGeneratorDlg::OnReleasedcaptureOutput(NMHDR* pNMHDR, LRESULT* pResult) 
   {
   UpdateValue();
   *pResult = 0;
   }

void CSelectGeneratorDlg::OnUpdateValue() 
   {
   UpdateData(true);
   m_mbGenerator(m_scInput.GetPos(), m_scOutput.GetPos()) = libbase::bitfield(m_sValue);
   }
