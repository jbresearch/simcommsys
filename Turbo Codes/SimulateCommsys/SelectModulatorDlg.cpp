// SelectModulatorDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SimulateCommsys.h"
#include "SelectModulatorDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSelectModulatorDlg dialog


CSelectModulatorDlg::CSelectModulatorDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CSelectModulatorDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CSelectModulatorDlg)
        m_nType = -1;
        //}}AFX_DATA_INIT
}


void CSelectModulatorDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CSelectModulatorDlg)
        DDX_CBIndex(pDX, IDC_TYPE, m_nType);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSelectModulatorDlg, CDialog)
        //{{AFX_MSG_MAP(CSelectModulatorDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSelectModulatorDlg message handlers
