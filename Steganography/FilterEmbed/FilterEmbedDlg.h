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

#ifndef afx_filterembeddlg_h
#define afx_filterembeddlg_h

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
// CFilterEmbedDlg dialog

class CFilterEmbedDlg : public CDialog
{
// Construction
public:
   CFilterEmbedDlg(CWnd* pParent = NULL);   // standard constructor
   libwin::CPSPlugIn*  m_pPSPlugIn;

// Dialog Data
   //{{AFX_DATA(CFilterEmbedDlg)
        enum { IDD = IDD_DIALOG1 };
        CString m_sCodec;
        CString m_sPuncture;
        CString m_sSource;
        int             m_nSourceSeed;
        int             m_nSourceType;
        BOOL    m_bInterleave;
        int             m_nInterleaverSeed;
        double  m_dInterleaverDensity;
        int             m_nEmbedSeed;
        double  m_dEmbedStrength;
        int             m_nEmbedRate;
        //}}AFX_DATA

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CFilterEmbedDlg)
   protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

// Implementation
protected:
   int m_nFileSize, m_nRawSize;
   int m_nCodecIn, m_nCodecOut;
   int m_nPunctureIn, m_nPunctureOut;

   void ComputeFileData();
   void ComputeCodecData();
   void ComputePunctureData();
        void UpdateDisplay();

   // Generated message map functions
   //{{AFX_MSG(CFilterEmbedDlg)
   virtual BOOL OnInitDialog();
        virtual void OnOK();
        afx_msg void OnLoadSource();
        afx_msg void OnLoadCodec();
        afx_msg void OnLoadPuncture();
        afx_msg void OnClearSource();
        afx_msg void OnClearCodec();
        afx_msg void OnClearPuncture();
        afx_msg void OnComputeStrength();
        afx_msg void OnInterleave();
        afx_msg void OnSelchangeSourceType();
        afx_msg void OnChangeInterleaverDensity();
        afx_msg void OnChangeEmbedRate();
        //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
