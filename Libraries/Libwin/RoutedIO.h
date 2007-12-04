#ifndef __routedio_h
#define __routedio_h

#include "MBoxStreamBuf.h"
#include "TraceStreamBuf.h"

/*
  Version 1.00 (10 Nov 2006)
  * Initial version - this class encapsulates the routing of standard I/O streams to
  the debug trace (for clog/cout) and to a message box (for cerr). This makes any
  derivative of this class automatically take over these facilities.
*/

namespace libwin {

class CRoutedIO
   {
   private:
      CMBoxStreamBuf    m_msgbox;
      CTraceStreamBuf   m_tracer;
   public:
      CRoutedIO();
   };

}; // end namespace

#endif
