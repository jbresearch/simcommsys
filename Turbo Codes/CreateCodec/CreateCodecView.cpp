// CreateCodecView.cpp : implementation of the CCreateCodecView class
//

#include "stdafx.h"
#include "CreateCodec.h"

#include "CreateCodecDoc.h"
#include "CreateCodecView.h"

#include "SelectCodecDlg.h"
#include "SelectEncoderDlg.h"
#include "SelectGeneratorDlg.h"
#include "SelectInterleaverDlg.h"
#include "SelectIntDlg.h"
#include "SelectBoolDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

typedef enum image_type {
   image_static = 0,
   image_codec,
   image_fsm,
   image_interleaver,
   image_generator,
   image_int,
   image_bool
   };

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecView

IMPLEMENT_DYNCREATE(CCreateCodecView, CTreeView)

BEGIN_MESSAGE_MAP(CCreateCodecView, CTreeView)
//{{AFX_MSG_MAP(CCreateCodecView)
        ON_NOTIFY_REFLECT(NM_DBLCLK, OnDblclk)
        //}}AFX_MSG_MAP
// Standard printing commands
ON_COMMAND(ID_FILE_PRINT, CTreeView::OnFilePrint)
ON_COMMAND(ID_FILE_PRINT_DIRECT, CTreeView::OnFilePrint)
ON_COMMAND(ID_FILE_PRINT_PREVIEW, CTreeView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecView construction/destruction

CCreateCodecView::CCreateCodecView()
   {
   // TODO: add construction code here

   }

CCreateCodecView::~CCreateCodecView()
   {
   }

BOOL CCreateCodecView::PreCreateWindow(CREATESTRUCT& cs)
   {
   // TODO: Modify the Window class or styles here by modifying
   //  the CREATESTRUCT cs
   cs.style |= TVS_HASLINES | TVS_LINESATROOT | TVS_HASBUTTONS;

   return CTreeView::PreCreateWindow(cs);
   }

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecView drawing

void CCreateCodecView::OnDraw(CDC* pDC)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   // TODO: add draw code for native data here
   }

void CCreateCodecView::OnInitialUpdate()
   {
   CTreeView::OnInitialUpdate();

   // TODO: You may populate your TreeView with items by directly accessing
   //  its tree control through a call to GetTreeCtrl().
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   GetTreeCtrl().InsertItem(pDoc->GetStringCodec(), image_codec, image_codec);
   }

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecView printing

BOOL CCreateCodecView::OnPreparePrinting(CPrintInfo* pInfo)
   {
   // default preparation
   return DoPreparePrinting(pInfo);
   }

void CCreateCodecView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
   {
   // TODO: add extra initialization before printing
   }

void CCreateCodecView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
   {
   // TODO: add cleanup after printing
   }

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecView diagnostics

#ifdef _DEBUG
void CCreateCodecView::AssertValid() const
   {
   CTreeView::AssertValid();
   }

void CCreateCodecView::Dump(CDumpContext& dc) const
   {
   CTreeView::Dump(dc);
   }

CCreateCodecDoc* CCreateCodecView::GetDocument() // non-debug version is inline
   {
   ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCreateCodecDoc)));
   return (CCreateCodecDoc*)m_pDocument;
   }
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecView message handlers

void CCreateCodecView::DeleteChildren(HTREEITEM hItem)
   {
   HTREEITEM hChild;
   while((hChild = GetTreeCtrl().GetChildItem(hItem)) != NULL)
      GetTreeCtrl().DeleteItem(hChild);
   }

void CCreateCodecView::OnDblclk(NMHDR* pNMHDR, LRESULT* pResult)
   {
   // Get position in client coordinates
   DWORD dwPos = ::GetMessagePos();
   CPoint point(LOWORD(dwPos), HIWORD(dwPos));
   GetTreeCtrl().ScreenToClient(&point);
   // Get item at that position & get its image type
   HTREEITEM hItem = GetTreeCtrl().HitTest(point);
   int nImage;
   GetTreeCtrl().GetItemImage(hItem, nImage, nImage);

   switch(nImage)
      {
      case image_static:
         break;

      case image_codec:
         SelectCodec(hItem);
         break;

      case image_fsm:
         SelectEncoder(hItem);
         break;

      case image_interleaver:
         SelectInterleaver(hItem);
         break;

      case image_generator:
         SelectGenerator(hItem);
         break;

      case image_int:
         SelectInt(hItem);
         break;

      case image_bool:
         SelectBool(hItem);
         break;

      default:
         CString sTemp;
         sTemp.Format("Unkown object type (%d).", nImage);
         MessageBox(sTemp);
         break;
      }

   // if the current item has children, make sure it's closed (this will be toggled later by system)
   if(GetTreeCtrl().GetChildItem(hItem) != NULL)
      GetTreeCtrl().Expand(hItem, TVE_COLLAPSE);

   *pResult = 0;
   }

void CCreateCodecView::SelectCodec(HTREEITEM hItem)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectCodecDlg dlg;
   dlg.m_nType = pDoc->GetCodecType();
   dlg.m_nMath = pDoc->GetArithmetic();
   if(dlg.DoModal() == IDOK)
      {
      if(pDoc->GetCodecType() == dlg.m_nType)
         {
         if(pDoc->GetArithmetic() != dlg.m_nMath)
            {
            pDoc->SetArithmetic(dlg.m_nMath);
            GetTreeCtrl().SetItemText(hItem, pDoc->GetStringCodec());
            }
         }
      else
         {
         pDoc->SetCodecType(dlg.m_nType);
         pDoc->SetArithmetic(dlg.m_nMath);
         GetTreeCtrl().SetItemText(hItem, pDoc->GetStringCodec());
         DeleteChildren(hItem);
         HTREEITEM hNewItem;
         switch(pDoc->GetCodecType())
            {
            case 0: // uncoded
            case 1: // mapcc
               GetTreeCtrl().InsertItem(pDoc->GetStringEncoder(), image_fsm, image_fsm, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringInt("block size"), image_int, image_int, hItem);
               break;
            case 2: // turbo
               GetTreeCtrl().InsertItem(pDoc->GetStringEncoder(), image_fsm, image_fsm, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringInt("block size"), image_int, image_int, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringInt("iterations"), image_int, image_int, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringBool("simile"), image_bool, image_bool, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringBool("terminated"), image_bool, image_bool, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringBool("parallel"), image_bool, image_bool, hItem);
               hNewItem = GetTreeCtrl().InsertItem("interleavers", image_static, image_static, hItem);
               GetTreeCtrl().InsertItem("<add>", image_interleaver, image_interleaver, hNewItem);
               break;
            case 3: // diffturbo
               break;
            }
         }
      }
   }

void CCreateCodecView::SelectEncoder(HTREEITEM hItem)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectEncoderDlg dlg;
   dlg.m_nType = pDoc->GetEncoderType();
   if(dlg.DoModal() == IDOK)
      {
      if(pDoc->GetEncoderType() != dlg.m_nType)
         {
         pDoc->SetEncoderType(dlg.m_nType);
         GetTreeCtrl().SetItemText(hItem, pDoc->GetStringEncoder());
         DeleteChildren(hItem);
         switch(pDoc->GetEncoderType())
            {
            case 0: // nrcc
            case 1: // rscc
               GetTreeCtrl().InsertItem(pDoc->GetStringInt("inputs"), image_int, image_int, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringInt("outputs"), image_int, image_int, hItem);
               GetTreeCtrl().InsertItem(pDoc->GetStringGenerator(), image_generator, image_generator, hItem);
               break;
            }
         }
      }
   }

void CCreateCodecView::SelectGenerator(HTREEITEM hItem)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectGeneratorDlg dlg;
   dlg.m_mbGenerator = pDoc->GetGenerator();
   if(dlg.DoModal() == IDOK)
      {
      pDoc->SetGenerator(dlg.m_mbGenerator);
      GetTreeCtrl().SetItemText(hItem, pDoc->GetStringGenerator());
      }
   }

void CCreateCodecView::SelectInterleaver(HTREEITEM hItem)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   CSelectInterleaverDlg dlg;
   if(dlg.DoModal() == IDOK)
      {
      pDoc->AddInterleaver(dlg.m_pInterleaver);
      GetTreeCtrl().InsertItem(pDoc->GetStringInterleaver(dlg.m_pInterleaver), image_interleaver, image_interleaver, GetTreeCtrl().GetParentItem(hItem));
      }
   }

void CCreateCodecView::SelectInt(HTREEITEM hItem)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   // find out what value we're trying to edit
   CString sTemp = GetTreeCtrl().GetItemText(hItem);
   CString sName = sTemp.Left(sTemp.Find(':'));

   CSelectIntDlg dlg;
   dlg.m_nValue = pDoc->GetIntValue(sName);
   if(dlg.DoModal() == IDOK)
      {
      pDoc->SetIntValue(sName, dlg.m_nValue);
      GetTreeCtrl().SetItemText(hItem, pDoc->GetStringInt(sName));
      }
   }

void CCreateCodecView::SelectBool(HTREEITEM hItem)
   {
   CCreateCodecDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);

   // find out what value we're trying to edit
   CString sTemp = GetTreeCtrl().GetItemText(hItem);
   CString sName = sTemp.Left(sTemp.Find(':'));

   // toggle its value and update display
   pDoc->SetBoolValue(sName, !pDoc->GetBoolValue(sName));
   GetTreeCtrl().SetItemText(hItem, pDoc->GetStringBool(sName));
   }
