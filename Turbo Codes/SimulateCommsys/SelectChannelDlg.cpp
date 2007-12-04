// SelectChannelDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SimulateCommsys.h"
#include "SelectChannelDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectChannelDlg dialog


CSelectChannelDlg::CSelectChannelDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CSelectChannelDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CSelectChannelDlg)
        m_nType = -1;
        //}}AFX_DATA_INIT
}


void CSelectChannelDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CSelectChannelDlg)
        DDX_CBIndex(pDX, IDC_TYPE, m_nType);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSelectChannelDlg, CDialog)
        //{{AFX_MSG_MAP(CSelectChannelDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectChannelDlg message handlers
