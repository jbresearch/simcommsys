#include "stdafx.h"
#include "filterlevels.h"
#include "DisplayResultsDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CDisplayResultsDlg dialog


CDisplayResultsDlg::CDisplayResultsDlg(CWnd* pParent /*=NULL*/)
        : CDialog(CDisplayResultsDlg::IDD, pParent)
{
        //{{AFX_DATA_INIT(CDisplayResultsDlg)
        m_sBlack = _T("");
        m_sWhite = _T("");
        //}}AFX_DATA_INIT
}


void CDisplayResultsDlg::DoDataExchange(CDataExchange* pDX)
{
        CDialog::DoDataExchange(pDX);
        //{{AFX_DATA_MAP(CDisplayResultsDlg)
        DDX_Text(pDX, IDC_BLACK, m_sBlack);
        DDX_Text(pDX, IDC_WHITE, m_sWhite);
        //}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CDisplayResultsDlg, CDialog)
        //{{AFX_MSG_MAP(CDisplayResultsDlg)
                // NOTE: the ClassWizard will add message map macros here
        //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CDisplayResultsDlg message handlers
