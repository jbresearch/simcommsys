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

#ifndef __workerthread_h
#define __workerthread_h

#include "StdAfx.h"

/*
   \version 1.10 (15 Nov 2001)
  moved <afxmt.h> into StdAfx.h, and modified ThreadStart to allow choosing a priority
  for the worker thread. Also, it now defaults to ThreadPriorityLowest rather than
  BelowNormal. This was instroduced because it was slowing down some programs (notably
  FileSync).

   \version 1.11 (25 Feb 2002)
  made ThreadWorking() and ThreadInterrupted() inline functions by defining them within
  the class desciption below. Also removed the ThreadProc() empty function definition,
  making it pure virtual, so that derived classes now have to provide their own (this
  class is intended to be used that way anyway). Also solved a logical bug in the
  ThreadAskAndWait() function - the question should have been "do you want to FORCE
  the thread to finish" since we're going to wait for it to finish anyway.

   \version 1.20 (26 Feb 2002)
  Removed the AskAndWait function - the derived class should handle this itself based
  on the available functions. Added functions to Suspend, Resume, and Kill the worker
  thread - of these, Kill is still not working properly;

   \version 1.30 (6 Nov 2006)
   - defined class and associated data within "libwin" namespace.
   - removed pragma once directive, as this is unnecessary
   - changed unique define to conform with that used in other libraries
*/

namespace libwin {

class CWorkerThread
{
private:
        static UINT ThreadProcRedirect(LPVOID pParam);
   bool m_bWorking;
   bool m_bInterrupted;
   CEvent m_eventDone;
   CWinThread *m_pThread;
protected:
        virtual void ThreadProc() = 0;
   bool ThreadWorking() const { return m_bWorking; };
   bool ThreadInterrupted() const { return m_bInterrupted; };
        void ThreadStart(int nPriority=THREAD_PRIORITY_LOWEST);
        void ThreadStop();
        void ThreadKill();
   void ThreadSuspend();
   void ThreadResume();
        void ThreadWaitFinish();
public:
        CWorkerThread();
        virtual ~CWorkerThread();
};

} // end namespace

#endif
