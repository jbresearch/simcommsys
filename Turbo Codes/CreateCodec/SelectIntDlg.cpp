// SelectIntDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "SelectIntDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectIntDlg dialog


CSelectIntDlg::CSelectIntDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CSelectIntDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CSelectIntDlg)
        m_nValue = 0;
        //}}AFX_DATA_INIT
}


void CSelectIntDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CSelectIntDlg)
        DDX_Text(pDX, IDC_VALUE, m_nValue);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSelectIntDlg, CDialog)
        //{{AFX_MSG_MAP(CSelectIntDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectIntDlg message handlers
