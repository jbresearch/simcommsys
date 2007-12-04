#if !defined(AFX_SELECTCHANNELDLG_H__71B96E4A_6E1E_4DF3_9A57_E16E1DEE7D34__INCLUDED_)
#define AFX_SELECTCHANNELDLG_H__71B96E4A_6E1E_4DF3_9A57_E16E1DEE7D34__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectChannelDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectChannelDlg dialog

class CSelectChannelDlg : public CDialog
{
// Construction
public:
        CSelectChannelDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectChannelDlg)
        enum { IDD = IDD_CHANNEL };
        int             m_nType;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectChannelDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectChannelDlg)
                // NOTE: the ClassWizard will add member functions here
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTCHANNELDLG_H__71B96E4A_6E1E_4DF3_9A57_E16E1DEE7D34__INCLUDED_)
