// SimulateCommsysView.cpp : implementation of the CSimulateCommsysView class
//

#include "stdafx.h"
#include "SimulateCommsys.h"

#include "SimulateCommsysDoc.h"
#include "SimulateCommsysView.h"

#include "SelectChannelDlg.h"
#include "SelectModulatorDlg.h"
#include "SelectAccuracyDlg.h"
#include "SelectRangeDlg.h"
#include "BenchmarkDlg.h"

#include "serializer_libcomm.h"
#include "commsys.h"

#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

const libcomm::serializer_libcomm g_serializer_libcomm;

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysView

IMPLEMENT_DYNCREATE(CSimulateCommsysView, CListView)

BEGIN_MESSAGE_MAP(CSimulateCommsysView, CListView)
//{{AFX_MSG_MAP(CSimulateCommsysView)
ON_COMMAND(ID_SYSTEM_CHANNEL, OnSystemChannel)
ON_COMMAND(ID_SYSTEM_MODULATION, OnSystemModulation)
	ON_UPDATE_COMMAND_UI(ID_SYSTEM_CHANNEL, OnUpdateSystemChannel)
	ON_UPDATE_COMMAND_UI(ID_SYSTEM_MODULATION, OnUpdateSystemModulation)
	ON_UPDATE_COMMAND_UI(ID_SYSTEM_PUNCTURING, OnUpdateSystemPuncturing)
	ON_UPDATE_COMMAND_UI(ID_SYSTEM_CODEC, OnUpdateSystemCodec)
	ON_COMMAND(ID_SYSTEM_PUNCTURING, OnSystemPuncturing)
	ON_COMMAND(ID_SYSTEM_CODEC, OnSystemCodec)
	ON_COMMAND(ID_SIMULATION_ACCURACY, OnSimulationAccuracy)
	ON_COMMAND(ID_SIMULATION_RANGE, OnSimulationRange)
	ON_COMMAND(ID_SIMULATION_START, OnSimulationStart)
	ON_COMMAND(ID_SIMULATION_STOP, OnSimulationStop)
	ON_UPDATE_COMMAND_UI(ID_SIMULATION_START, OnUpdateSimulationStart)
	ON_UPDATE_COMMAND_UI(ID_SIMULATION_STOP, OnUpdateSimulationStop)
	ON_UPDATE_COMMAND_UI(ID_SIMULATION_ACCURACY, OnUpdateSimulationAccuracy)
	ON_UPDATE_COMMAND_UI(ID_SIMULATION_RANGE, OnUpdateSimulationRange)
	ON_UPDATE_COMMAND_UI(ID_FILE_SAVE, OnUpdateFileSave)
	ON_UPDATE_COMMAND_UI(ID_FILE_SAVE_AS, OnUpdateFileSaveAs)
	ON_COMMAND(ID_SIMULATION_BENCHMARK, OnSimulationBenchmark)
	ON_UPDATE_COMMAND_UI(ID_SIMULATION_BENCHMARK, OnUpdateSimulationBenchmark)
	//}}AFX_MSG_MAP
// Standard printing commands
ON_COMMAND(ID_FILE_PRINT, CListView::OnFilePrint)
ON_COMMAND(ID_FILE_PRINT_DIRECT, CListView::OnFilePrint)
ON_COMMAND(ID_FILE_PRINT_PREVIEW, CListView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysView construction/destruction

CSimulateCommsysView::CSimulateCommsysView()
   {
   // TODO: add construction code here
   m_dSNRmin = 0.0;
   m_dSNRmax = 5.0;
   m_dSNRstep = 0.5;
   }

CSimulateCommsysView::~CSimulateCommsysView()
   {
   }

BOOL CSimulateCommsysView::PreCreateWindow(CREATESTRUCT& cs)
   {
   // TODO: Modify the Window class or styles here by modifying
   //  the CREATESTRUCT cs
   cs.style &= ~LVS_TYPEMASK;
   cs.style |= LVS_REPORT;
   
   return CListView::PreCreateWindow(cs);
   }

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysView drawing

void CSimulateCommsysView::OnDraw(CDC* pDC)
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   // TODO: add draw code for native data here
   }

void CSimulateCommsysView::OnInitialUpdate()
   {
   CListView::OnInitialUpdate();
   
   // TODO: You may populate your ListView with items by directly accessing
   //  its list control through a call to GetListCtrl().

   // delete all items and columns from the list
   GetListCtrl().DeleteAllItems();
   while(GetListCtrl().DeleteColumn(0))
      ;
   // set up visuals
   GetListCtrl().InsertColumn(0, "SNR", LVCFMT_CENTER, 50, 0);
   GetListCtrl().InsertColumn(1, "Frames", LVCFMT_CENTER, 50, 1);
   GetListCtrl().InsertColumn(2, "Duration", LVCFMT_CENTER, 100, 2);
   }

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysView printing

BOOL CSimulateCommsysView::OnPreparePrinting(CPrintInfo* pInfo)
   {
   // default preparation
   return DoPreparePrinting(pInfo);
   }

void CSimulateCommsysView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
   {
   // TODO: add extra initialization before printing
   }

void CSimulateCommsysView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
   {
   // TODO: add cleanup after printing
   }

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysView diagnostics

#ifdef _DEBUG
void CSimulateCommsysView::AssertValid() const
   {
   CListView::AssertValid();
   }

void CSimulateCommsysView::Dump(CDumpContext& dc) const
   {
   CListView::Dump(dc);
   }

CSimulateCommsysDoc* CSimulateCommsysView::GetDocument() // non-debug version is inline
   {
   ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CSimulateCommsysDoc)));
   return (CSimulateCommsysDoc*)m_pDocument;
   }
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysView message handlers

void CSimulateCommsysView::OnSystemChannel() 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectChannelDlg dlg;
   if(dlg.DoModal() == IDOK)
      {
      switch(dlg.m_nType)
         {
         case 0:
            pDoc->SetChannel(new libcomm::awgn);
            break;
         case 1:
            pDoc->SetChannel(new libcomm::laplacian);
            break;
         }
      }
   }

void CSimulateCommsysView::OnSystemModulation() 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectModulatorDlg dlg;
   if(dlg.DoModal() == IDOK)
      {
      switch(dlg.m_nType)
         {
         case 0:
            pDoc->SetModulator(new libcomm::mpsk(2));
            break;
         case 1:
            pDoc->SetModulator(new libcomm::mpsk(4));
            break;
         }
      }
   }

void CSimulateCommsysView::OnSystemPuncturing() 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CFileDialog dlg(true, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      std::ifstream file(dlg.GetPathName());
      libcomm::puncture *pPuncture;
      file >> pPuncture;
      pDoc->SetPuncture(pPuncture);
      }
   }

void CSimulateCommsysView::OnSystemCodec() 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CFileDialog dlg(true, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      std::ifstream file(dlg.GetPathName());
      libcomm::codec *pCodec;
      file >> pCodec;
      pDoc->SetCodec(pCodec);
      }
   }

void CSimulateCommsysView::OnSimulationAccuracy() 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectAccuracyDlg dlg;
   dlg.m_dAccuracy = pDoc->GetAccuracy();
   dlg.m_dConfidence = pDoc->GetConfidence();
   if(dlg.DoModal() == IDOK)
      {
      pDoc->SetAccuracy(dlg.m_dAccuracy);
      pDoc->SetConfidence(dlg.m_dConfidence);
      }
   }

void CSimulateCommsysView::OnSimulationRange() 
   {
   CSelectRangeDlg dlg;
   dlg.m_dSNRmin = m_dSNRmin;
   dlg.m_dSNRmax = m_dSNRmax;
   dlg.m_dSNRstep = m_dSNRstep;
   if(dlg.DoModal() == IDOK)
      {
      m_dSNRmin = dlg.m_dSNRmin;
      m_dSNRmax = dlg.m_dSNRmax;
      m_dSNRstep = dlg.m_dSNRstep;
      }
   }

void CSimulateCommsysView::OnSimulationStart() 
   {
   ThreadStart();
   }

void CSimulateCommsysView::OnSimulationStop() 
   {
   ThreadStop();
   }

void CSimulateCommsysView::OnSimulationBenchmark() 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CBenchmarkDlg dlg;
   dlg.m_pCodec = pDoc->GetCodec();
   dlg.m_pPuncture = pDoc->GetPuncture();
   dlg.m_pModulator = pDoc->GetModulator();
   dlg.m_pChannel = pDoc->GetChannel();
   dlg.m_dSNR = m_dSNRmin;
   dlg.m_dTime = 60;
   dlg.DoModal();
   }

// menu update functions

void CSimulateCommsysView::OnUpdateSystemChannel(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->SetCheck(pDoc->GetChannel() != NULL);
   pCmdUI->Enable(!ThreadWorking() && !pDoc->ResultsPresent());
   }

void CSimulateCommsysView::OnUpdateSystemModulation(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->SetCheck(pDoc->GetModulator() != NULL);
   pCmdUI->Enable(!ThreadWorking() && !pDoc->ResultsPresent());
   }

void CSimulateCommsysView::OnUpdateSystemPuncturing(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->SetCheck(pDoc->GetPuncture() != NULL);
   pCmdUI->Enable(!ThreadWorking() && !pDoc->ResultsPresent());
   }

void CSimulateCommsysView::OnUpdateSystemCodec(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->SetCheck(pDoc->GetCodec() != NULL);
   pCmdUI->Enable(!ThreadWorking() && !pDoc->ResultsPresent());
   }

void CSimulateCommsysView::OnUpdateSimulationAccuracy(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->Enable(!ThreadWorking() && !pDoc->ResultsPresent());
   }

void CSimulateCommsysView::OnUpdateSimulationRange(CCmdUI* pCmdUI) 
   {
   pCmdUI->Enable(!ThreadWorking());
   }

void CSimulateCommsysView::OnUpdateSimulationStart(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->Enable(!ThreadWorking() && pDoc->GetCodec() != NULL && pDoc->GetModulator() != NULL && pDoc->GetChannel() != NULL);
   }

void CSimulateCommsysView::OnUpdateSimulationStop(CCmdUI* pCmdUI) 
   {
   pCmdUI->Enable(ThreadWorking());
   }

void CSimulateCommsysView::OnUpdateSimulationBenchmark(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->Enable(!ThreadWorking() && pDoc->GetCodec() != NULL && pDoc->GetModulator() != NULL && pDoc->GetChannel() != NULL);
   }

void CSimulateCommsysView::OnUpdateFileSave(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->Enable(pDoc->ResultsPresent());
   }

void CSimulateCommsysView::OnUpdateFileSaveAs(CCmdUI* pCmdUI) 
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   pCmdUI->Enable(pDoc->ResultsPresent());
   }

// internal functions

void CSimulateCommsysView::InsertResults(const double dSNR, const int iSamples, const libbase::vector<double>& vdEstimate, const libbase::vector<double>& vdError, const double dElapsed)
   {
   CString sTemp;
   LVITEM item;
   item.mask = LVIF_TEXT;
   item.iItem = GetListCtrl().GetItemCount();

   sTemp.Format("%g", dSNR);
   item.iSubItem = 0;
   item.pszText = LPTSTR(LPCTSTR(sTemp));
   GetListCtrl().InsertItem(&item);

   sTemp.Format("%d", iSamples);
   item.iSubItem = 1;
   item.pszText = LPTSTR(LPCTSTR(sTemp));
   GetListCtrl().SetItem(&item);

   sTemp.Format("%s", libbase::timer::format(dElapsed).c_str());
   item.iSubItem = 2;
   item.pszText = LPTSTR(LPCTSTR(sTemp));
   GetListCtrl().SetItem(&item);

   for(int i=0; i<vdEstimate.size(); i++)
      {
      sTemp.Format("Result %d", i);
      if(item.iItem == 0)
         GetListCtrl().InsertColumn(i+3, sTemp, LVCFMT_CENTER, 100, i+3);
      sTemp.Format("%0.4g ± %2.1f%%", vdEstimate(i), 100*vdError(i));
      item.iSubItem = i+3;
      item.pszText = LPTSTR(LPCTSTR(sTemp));
      GetListCtrl().SetItem(&item);
      }

   GetListCtrl().EnsureVisible(item.iItem, false);
   }

// montecarlo overrides

void CSimulateCommsysView::display(const int pass, const double cur_accuracy, const double cur_mean)
   {
   /*
   CStatusGraph::Insert(GetDlgItem(IDC_GRAPH1), cur_accuracy);
   CStatusGraph::Insert(GetDlgItem(IDC_GRAPH2), cur_mean);

   CString str;

   str.Format("%d", pass);
   GetDlgItem(IDC_DISP_PASSES)->SetWindowText(str);
   str.Format("%0.1f%%", cur_accuracy);
   GetDlgItem(IDC_DISP_ACCURACY)->SetWindowText(str);
   str.Format("%0.4g", cur_mean);
   GetDlgItem(IDC_DISP_MEAN)->SetWindowText(str);

   m_pcCurrent.SetPos(int(floor(100 - 100 * log(cur_accuracy / m_nAccuracy) / log(100 / m_nAccuracy) )));
   */
   CSimulateCommsysDoc* pDoc = GetDocument();
   m_dProgress = 100 - 100 * log(cur_accuracy / pDoc->GetAccuracy()) / log(100 / pDoc->GetAccuracy());
   }

// WorkerThread overrides

void CSimulateCommsysView::ThreadProc()
   {
   CSimulateCommsysDoc* pDoc = GetDocument();
   ASSERT(pDoc->GetCodec() != NULL);
   ASSERT(pDoc->GetModulator() != NULL);
   ASSERT(pDoc->GetChannel() != NULL);

   // Source Generator
   libbase::randgen src;
   // The complete communication system
   libcomm::commsys system(&src, pDoc->GetCodec(), pDoc->GetModulator(), pDoc->GetPuncture(), pDoc->GetChannel());
   // The actual estimator - tie to the system and set up
   initialise(&system);
   set_confidence(pDoc->GetConfidence());
   set_accuracy(pDoc->GetAccuracy());

   // Work out the following for every SNR value required
   libbase::timer tSimulation;
   for(m_dSNR=m_dSNRmin; m_dSNR<=m_dSNRmax && !ThreadInterrupted(); m_dSNR+=m_dSNRstep)
      {
      tSimulation.start();
      pDoc->GetChannel()->set_snr(m_dSNR);
      libbase::vector<double> est, tol;
      estimate(est, tol);
      tSimulation.stop();
      
      pDoc->InsertResults(m_dSNR, get_samplecount(), est, tol, tSimulation.elapsed());
      InsertResults(m_dSNR, get_samplecount(), est, tol, tSimulation.elapsed());
      }

   // release the estimator so we can use it again
   finalise();
   }
