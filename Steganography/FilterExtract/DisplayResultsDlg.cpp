#include "stdafx.h"
#include "filterextract.h"
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
	m_sBER = _T("");
	m_sRate = _T("");
	m_sSNR = _T("");
	m_sSNRest = _T("");
	m_sChiSquare = _T("");
	//}}AFX_DATA_INIT
}


void CDisplayResultsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CDisplayResultsDlg)
	DDX_Text(pDX, IDC_BER, m_sBER);
	DDX_Text(pDX, IDC_RATE, m_sRate);
	DDX_Text(pDX, IDC_SNR, m_sSNR);
	DDX_Text(pDX, IDC_SNR_EST, m_sSNRest);
	DDX_Text(pDX, IDC_CHI_SQUARE, m_sChiSquare);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CDisplayResultsDlg, CDialog)
	//{{AFX_MSG_MAP(CDisplayResultsDlg)
		// NOTE: the ClassWizard will add message map macros here
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CDisplayResultsDlg message handlers
