// SelectEncoderDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "SelectEncoderDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectEncoderDlg dialog


CSelectEncoderDlg::CSelectEncoderDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CSelectEncoderDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CSelectEncoderDlg)
        m_nType = -1;
        //}}AFX_DATA_INIT
}


void CSelectEncoderDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CSelectEncoderDlg)
        DDX_CBIndex(pDX, IDC_TYPE, m_nType);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSelectEncoderDlg, CDialog)
        //{{AFX_MSG_MAP(CSelectEncoderDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectEncoderDlg message handlers
