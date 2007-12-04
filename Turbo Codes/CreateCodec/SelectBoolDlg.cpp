// SelectBoolDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "SelectBoolDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectBoolDlg dialog


CSelectBoolDlg::CSelectBoolDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CSelectBoolDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CSelectBoolDlg)
        m_nValue = -1;
        //}}AFX_DATA_INIT
}


void CSelectBoolDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CSelectBoolDlg)
        DDX_Radio(pDX, IDC_NAY, m_nValue);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSelectBoolDlg, CDialog)
        //{{AFX_MSG_MAP(CSelectBoolDlg)
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectBoolDlg message handlers

