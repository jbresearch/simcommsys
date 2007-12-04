#include "stdafx.h"
#include "RoutedIO.h"
#include <iostream>

namespace libwin {

CRoutedIO::CRoutedIO()
   {
   // route standard streams to trace/message box output
   std::ostream tracer(&m_tracer);
   std::ostream msgbox(&m_msgbox);
   std::clog.rdbuf(tracer.rdbuf());
   std::cout.rdbuf(tracer.rdbuf());
   std::cerr.rdbuf(msgbox.rdbuf());
   }

}; // end namespace
