#include "stdafx.h"
#include "Histogram2D.h"
#include <math.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

namespace libwin {

LRESULT CALLBACK AFX_EXPORT CHistogram2DWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
   {
   AFX_MANAGE_STATE(AfxGetStaticModuleState());

   CWnd* pWnd;

   pWnd = CWnd::FromHandlePermanent(hWnd);
   if (pWnd == NULL) {
      // Assume that client created a CHistogram2D window
      pWnd = new CHistogram2D();
      pWnd->Attach(hWnd);
      }
   ASSERT(pWnd->m_hWnd == hWnd);
   ASSERT(pWnd == CWnd::FromHandlePermanent(hWnd));
   LRESULT lResult = AfxCallWndProc(pWnd, hWnd, message, wParam, lParam);
   return lResult;
   }

/////////////////////////////////////////////////////////////////////////////
// CHistogram2D

CHistogram2D::CHistogram2D() : m_data(0,0)
   {
   TRACE("CHistogram2D constructor\n");
   }

CHistogram2D::~CHistogram2D()
   {
   TRACE("CHistogram2D destructor\n");
   }


BEGIN_MESSAGE_MAP(CHistogram2D, CWnd)
   //{{AFX_MSG_MAP(CHistogram2D)
   ON_WM_PAINT()
   //}}AFX_MSG_MAP
   //ON_MESSAGE(WM_USER, OnUser)
END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// CHistogram2D message handlers

void CHistogram2D::OnPaint()
   {
   CPaintDC dc(this); // device context for painting

   CRect rectClient;
   GetClientRect(rectClient);
   const int xsize = rectClient.Width();
   const int ysize = rectClient.Height();

   const double sg = 255/double(m_maxval);
   const double sx = xsize/double(m_data.xsize());
   const double sy = ysize/double(m_data.ysize());
   const int cx = int(ceil(sx));
   const int cy = int(ceil(sy));

   for(int i=0; i<m_data.xsize(); i++)
      for(int j=0; j<m_data.ysize(); j++)
         {
         const int g = int(floor(m_data(i,j) * sg));
         const int c = g | g<<8 | g<<16;
         const int x = int(floor(i * sx));
         const int y = int(floor(j * sy));
         dc.FillSolidRect(x,ysize-y,cx,-cy, c);
         }

   CRect rectWindow;
   GetWindowRect(rectWindow);
   dc.DrawEdge(rectWindow,EDGE_SUNKEN,BF_RECT);
   // Do not call CWnd::OnPaint() for painting messages
   }

void CHistogram2D::OnUser(WPARAM wParam, LPARAM lParam)
   {
   using libbase::trace;
   libbase::matrix<int> *data = (libbase::matrix<int> *) wParam;
   trace << "DEBUG (Histogram2D): received matrix at " << data << "\n";
   m_data = *data;
   trace << "DEBUG (Histogram2D): copied matrix " << m_data.xsize() << "x" << m_data.ysize() << "\n";
   m_maxval = m_data.max();
   trace << "DEBUG (Histogram2D): max value = " << m_maxval << "\n";
   Invalidate(false);
   }

bool CHistogram2D::RegisterWndClass(HINSTANCE hInstance)
   {
   WNDCLASS wc;
   wc.lpszClassName = "Histogram2D";
   wc.hInstance = hInstance;
   wc.lpfnWndProc = CHistogram2DWndProc;
   wc.hCursor = ::LoadCursor(NULL, IDC_ARROW);
   wc.hIcon = 0;
   wc.lpszMenuName = NULL;
   wc.hbrBackground = (HBRUSH) ::GetStockObject(WHITE_BRUSH);
   wc.style = CS_GLOBALCLASS;
   wc.cbClsExtra = 0;
   wc.cbWndExtra = 0;
   return (::RegisterClass(&wc) != 0);
   }

void CHistogram2D::UpdateData(CWnd *pWnd, libbase::matrix<int>& m)
   {
   pWnd->SendMessage(WM_USER, (UINT_PTR) &m);
   }

}; // end namespace
