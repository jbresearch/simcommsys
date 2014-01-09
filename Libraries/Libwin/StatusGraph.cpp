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

#include "stdafx.h"
#include "StatusGraph.h"
#include <afxdlgs.h>
#include <fstream>
#include <math.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

namespace libwin {

LRESULT CALLBACK AFX_EXPORT CStatusGraphWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
   {
   AFX_MANAGE_STATE(AfxGetStaticModuleState());

   CWnd* pWnd;

   pWnd = CWnd::FromHandlePermanent(hWnd);
   if (pWnd == NULL) {
      // Assume that client created a CStatusGraph window
      pWnd = new CStatusGraph();
      pWnd->Attach(hWnd);
      }
   ASSERT(pWnd->m_hWnd == hWnd);
   ASSERT(pWnd == CWnd::FromHandlePermanent(hWnd));
   LRESULT lResult = AfxCallWndProc(pWnd, hWnd, message, wParam, lParam);
   return lResult;
   }

/////////////////////////////////////////////////////////////////////////////
// CStatusGraph

CStatusGraph::CStatusGraph()
   {
   }

CStatusGraph::~CStatusGraph()
   {
   }


BEGIN_MESSAGE_MAP(CStatusGraph, CWnd)
        //{{AFX_MSG_MAP(CStatusGraph)
        ON_WM_PAINT()
   ON_WM_LBUTTONDOWN()
        //}}AFX_MSG_MAP
   //ON_MESSAGE(WM_STATUSGRAPH_RESET, OnReset)
   //ON_MESSAGE(WM_STATUSGRAPH_INSERT, OnInsert)
END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// CStatusGraph message handlers

void CStatusGraph::OnPaint()
   {
        CPaintDC dc(this); // device context for painting

   CRect rectClient;
   GetClientRect(rectClient);
   const int xsize = rectClient.Width();
   const int ysize = rectClient.Height();

   dc.SelectStockObject(BLACK_BRUSH);

   POSITION p = m_data.GetHeadPosition();
   if(p != NULL)
      dc.MoveTo(0, int(floor(ysize-1 - (ysize-1) * m_data.GetHead()/m_maxval)));
   for(int x=0; p!=NULL && x<xsize; x++)
      dc.LineTo(x, int(floor(ysize-1 - (ysize-1) * m_data.GetNext(p)/m_maxval)));

   CRect rectWindow;
   GetWindowRect(rectWindow);
   dc.DrawEdge(rectWindow,EDGE_SUNKEN,BF_RECT);
        // Do not call CWnd::OnPaint() for painting messages
   }

void CStatusGraph::OnLButtonDown(UINT nFlags, CPoint point)
   {
   CFileDialog dlg(FALSE, "txt", "*.txt");
   if(dlg.DoModal() == IDOK)
      {
      std::ofstream file(dlg.GetPathName());

      POSITION p = m_data.GetHeadPosition();
      while(p != NULL)
         file << m_data.GetNext(p) << "\n";

      file.close();
      }
   }

void CStatusGraph::OnReset(WPARAM wParam, LPARAM lParam)
   {
   m_data.RemoveAll();
   m_maxval = 0;
   Invalidate(true);
   }

void CStatusGraph::OnInsert(WPARAM wParam, LPARAM lParam)
   {
   double *x = (double *) wParam;
   m_data.AddHead(*x);
   if(m_maxval < *x)
      m_maxval = *x;
   Invalidate(true);
   delete x;
   }

bool CStatusGraph::RegisterWndClass(HINSTANCE hInstance)
   {
   WNDCLASS wc;
   wc.lpszClassName = "StatusGraph";
   wc.hInstance = hInstance;
   wc.lpfnWndProc = CStatusGraphWndProc;
   wc.hCursor = ::LoadCursor(NULL, IDC_ARROW);
   wc.hIcon = 0;
   wc.lpszMenuName = NULL;
   wc.hbrBackground = (HBRUSH) ::GetStockObject(WHITE_BRUSH);
   wc.style = CS_GLOBALCLASS;
   wc.cbClsExtra = 0;
   wc.cbWndExtra = 0;
   return (::RegisterClass(&wc) != 0);
   }

void CStatusGraph::Reset(CWnd *pWnd)
   {
   pWnd->SendMessage(WM_STATUSGRAPH_RESET);
   }

void CStatusGraph::Insert(CWnd *pWnd, const double x)
   {
   pWnd->SendMessage(WM_STATUSGRAPH_INSERT, (UINT_PTR) new double(x));
   }

} // end namespace
