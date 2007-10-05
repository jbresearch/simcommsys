#include "stdafx.h"
#include "filterextract.h"
#include "ComputeStrengthDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CComputeStrengthDlg dialog


CComputeStrengthDlg::CComputeStrengthDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CComputeStrengthDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CComputeStrengthDlg)
	m_dPower = 0.0;
	//}}AFX_DATA_INIT
}


void CComputeStrengthDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CComputeStrengthDlg)
	DDX_Text(pDX, IDC_POWER, m_dPower);
	DDV_MinMaxDouble(pDX, m_dPower, 0., 65025.);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CComputeStrengthDlg, CDialog)
	//{{AFX_MSG_MAP(CComputeStrengthDlg)
		// NOTE: the ClassWizard will add message map macros here
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CComputeStrengthDlg message handlers
