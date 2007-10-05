#ifndef __tracestreambuf_h
#define __tracestreambuf_h

#include <iostream>
#include <ios>

/*
  Version 1.10 (6 Nov 2006)
  * defined class and associated data within "libwin" namespace.
  * removed pragma once directive, as this is unnecessary
  * changed unique define to conform with that used in other libraries
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libwin {

class CTraceStreamBuf : public std::streambuf
{
protected:
   CString buffer;
public:
	CTraceStreamBuf();
	virtual ~CTraceStreamBuf();
   int underflow();
   int overflow(int nCh = EOF);
};

inline int CTraceStreamBuf::underflow()
   {
   return EOF;
   }

inline int CTraceStreamBuf::overflow(int nCh)
   {
   if(nCh=='\r' || nCh=='\n')
      {
      if(!buffer.IsEmpty())
         TRACE("%s\n", buffer);
      buffer = "";
      }
   else if(buffer.GetLength() == 250)
      {
      TRACE("%s", buffer);
      buffer = "";
      }
   else
      buffer += char(nCh);
   return 1;
   };

}; // end namespace

#endif
