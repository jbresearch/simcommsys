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
