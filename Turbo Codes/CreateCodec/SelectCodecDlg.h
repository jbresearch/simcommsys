#if !defined(AFX_SELECTCODECDLG_H__FDC95C9C_C4A1_4B36_9B3F_FCF4A06848B9__INCLUDED_)
#define AFX_SELECTCODECDLG_H__FDC95C9C_C4A1_4B36_9B3F_FCF4A06848B9__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// SelectCodecDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CSelectCodecDlg dialog

class CSelectCodecDlg : public CDialog
{
// Construction
public:
        CSelectCodecDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
        //{{AFX_DATA(CSelectCodecDlg)
        enum { IDD = IDD_CODEC };
        int             m_nMath;
        int             m_nType;
        //}}AFX_DATA


// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSelectCodecDlg)
        protected:
        virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
        //}}AFX_VIRTUAL

// Implementation
protected:

        // Generated message map functions
        //{{AFX_MSG(CSelectCodecDlg)
        afx_msg void OnSelchangeType();
        virtual BOOL OnInitDialog();
        virtual void OnOK();
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SELECTCODECDLG_H__FDC95C9C_C4A1_4B36_9B3F_FCF4A06848B9__INCLUDED_)
