#include "stdafx.h"
#include "WorkerThread.h"

namespace libwin {

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CWorkerThread::CWorkerThread()
   {
   m_bWorking = false;
   m_bInterrupted = false;
   }

CWorkerThread::~CWorkerThread()
   {
   }

//////////////////////////////////////////////////////////////////////
// Static Thread-Process Function
//////////////////////////////////////////////////////////////////////

UINT CWorkerThread::ThreadProcRedirect(LPVOID pParam)
   {
   CWorkerThread *obj = (CWorkerThread *)pParam;
   obj->m_eventDone.ResetEvent();
   obj->m_bWorking = true;
   obj->m_bInterrupted = false;;
   obj->ThreadProc();
   obj->m_bWorking = false;
   obj->m_eventDone.SetEvent();
   return 0;
   }

//////////////////////////////////////////////////////////////////////
// Thread-Control Functions (to be used by controlling object)
//////////////////////////////////////////////////////////////////////

void CWorkerThread::ThreadStart(int nPriority)
   {
   m_pThread = AfxBeginThread(ThreadProcRedirect, this, nPriority);
   }

void CWorkerThread::ThreadStop()
   {
   if(m_bWorking)
      m_bInterrupted = true;;
   }

void CWorkerThread::ThreadKill()
   {
   if(m_bWorking)
      ::TerminateThread(m_pThread->m_hThread, 0);
   }

void CWorkerThread::ThreadSuspend()
   {
   if(m_bWorking)
      m_pThread->SuspendThread();
   }

void CWorkerThread::ThreadResume()
   {
   if(m_bWorking)
      m_pThread->ResumeThread();
   }

void CWorkerThread::ThreadWaitFinish()
   {
   if(m_bWorking)
      ::WaitForSingleObject(m_eventDone, INFINITE);
   }

} // end namespace
