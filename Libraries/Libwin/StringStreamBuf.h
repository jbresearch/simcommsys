#ifndef __stringstream_h
#define __stringstream_h

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

class CStringStreamBuf : public std::streambuf  
{
protected:
   CString *buffer;
public:
	CStringStreamBuf(CString *string);
	virtual ~CStringStreamBuf();
   int underflow();
   int overflow(int nCh = EOF);
};

inline int CStringStreamBuf::underflow()
   {
   return EOF;
   }

inline int CStringStreamBuf::overflow(int nCh)
   {
   *buffer += (nCh == '\r') ? '\n' : char(nCh);
   return 1;
   };

}; // end namespace

#endif
