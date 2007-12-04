// SelectInterleaverDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "SelectInterleaverDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectInterleaverDlg dialog


CSelectInterleaverDlg::CSelectInterleaverDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CSelectInterleaverDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CSelectInterleaverDlg)
        m_nType = -1;
        //}}AFX_DATA_INIT
}


void CSelectInterleaverDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CSelectInterleaverDlg)
        DDX_CBIndex(pDX, IDC_TYPE, m_nType);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSelectInterleaverDlg, CDialog)
        //{{AFX_MSG_MAP(CSelectInterleaverDlg)
        ON_CBN_SELCHANGE(IDC_TYPE, OnSelchangeType)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectInterleaverDlg message handlers

void CSelectInterleaverDlg::OnSelchangeType() 
   {
   }
