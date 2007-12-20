// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "SimulateCommsys.h"

#include "MainFrm.h"
#include "SimulateCommsysDoc.h"
#include "SimulateCommsysView.h"

#include <sstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
//{{AFX_MSG_MAP(CMainFrame)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code !
ON_WM_CREATE()
//}}AFX_MSG_MAP
ON_UPDATE_COMMAND_UI(ID_INDICATOR_WORKING, OnUpdateWorking)
ON_UPDATE_COMMAND_UI(ID_INDICATOR_SNR, OnUpdateSNR)
ON_UPDATE_COMMAND_UI(ID_INDICATOR_PROGRESS, OnUpdateProgress)
END_MESSAGE_MAP()

static UINT indicators[] =
   {
   ID_SEPARATOR,  // status line indicator
   ID_INDICATOR_WORKING,
   ID_INDICATOR_SNR,
   ID_INDICATOR_PROGRESS,
   };

/////////////////////////////////////////////////////////////////////////////
// CMainFrame construction/destruction

CMainFrame::CMainFrame()
   {
   }

CMainFrame::~CMainFrame()
   {
   }

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
   {
   if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
      return -1;

   if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP
      | CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
      !m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
      {
      TRACE0("Failed to create toolbar\n");
      return -1;      // fail to create
      }

   if (!m_wndStatusBar.Create(this) ||
      !m_wndStatusBar.SetIndicators(indicators,
      sizeof(indicators)/sizeof(UINT)))
      {
      TRACE0("Failed to create status bar\n");
      return -1;      // fail to create
      }

   // TODO: Delete these three lines if you don't want the toolbar to
   //  be dockable
   m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
   EnableDocking(CBRS_ALIGN_ANY);
   DockControlBar(&m_wndToolBar);

   return 0;
   }

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
   {
   if( !CFrameWnd::PreCreateWindow(cs) )
      return FALSE;
   // TODO: Modify the Window class or styles here by modifying
   //  the CREATESTRUCT cs

   return TRUE;
   }

/////////////////////////////////////////////////////////////////////////////
// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
   {
   CFrameWnd::AssertValid();
   }

void CMainFrame::Dump(CDumpContext& dc) const
   {
   CFrameWnd::Dump(dc);
   }

#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CMainFrame message handlers

void CMainFrame::OnUpdateWorking(CCmdUI* pCmdUI)
   {
   CSimulateCommsysView* pView = (CSimulateCommsysView*) GetActiveView();
   if(pView != NULL)
      pCmdUI->SetText(pView->Working() ? "\tWorking" : "\tIdle");
   else
      pCmdUI->Enable(false);
   }

void CMainFrame::OnUpdateSNR(CCmdUI* pCmdUI)
   {
   CSimulateCommsysView* pView = (CSimulateCommsysView*) GetActiveView();
   if(pView != NULL)
      {
      pCmdUI->Enable(pView->Working());
      CString sTemp;
      sTemp.Format("\tSNR = %g", pView->GetSNR());
      pCmdUI->SetText(sTemp);
      }
   else
      pCmdUI->Enable(false);
   }

void CMainFrame::OnUpdateProgress(CCmdUI* pCmdUI)
   {
   CSimulateCommsysView* pView = (CSimulateCommsysView*) GetActiveView();
   if(pView != NULL)
      {
      pCmdUI->Enable(pView->Working());
      CString sTemp;
      sTemp.Format("\t%2.1f%%", pView->GetProgress());
      pCmdUI->SetText(sTemp);
      }
   else
      pCmdUI->Enable(false);
   }
