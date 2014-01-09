/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef afx_automategraphingdlg_h
#define afx_automategraphingdlg_h

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
   CAboutDlg();

   // Dialog Data
   //{{AFX_DATA(CAboutDlg)
   enum { IDD = IDD_ABOUTBOX };
   //}}AFX_DATA

   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAboutDlg)
protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

   // Implementation
protected:
   //{{AFX_MSG(CAboutDlg)
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////
// CAutomateGraphingDlg dialog

class CAutomateGraphingDlg : public CDialog
{
// Construction
public:
   CAutomateGraphingDlg(CWnd* pParent = NULL);   // standard constructor

   libwin::CPSAutomate*  m_pPSAutomate;

// Dialog Data
   //{{AFX_DATA(CAutomateGraphingDlg)
        enum { IDD = IDD_DIALOG1 };
        int             m_nJpegMin;
        int             m_nJpegMax;
        double  m_dStrengthMax;
        double  m_dStrengthMin;
        BOOL    m_bJpeg;
        int             m_nJpegStep;
        double  m_dStrengthStep;
        CString m_sParameters;
        CString m_sResults;
        BOOL    m_bPresetStrength;
        BOOL    m_bPrintBER;
        BOOL    m_bPrintChiSquare;
        BOOL    m_bPrintEstimate;
        BOOL    m_bPrintSNR;
        //}}AFX_DATA

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAutomateGraphingDlg)
   protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

// Implementation
protected:

   // Generated message map functions
   //{{AFX_MSG(CAutomateGraphingDlg)
   virtual BOOL OnInitDialog();
        afx_msg void OnParametersBrowse();
        afx_msg void OnResultsBrowse();
        afx_msg void OnJpeg();
        virtual void OnOK();
        //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif


