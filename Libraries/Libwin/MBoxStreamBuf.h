#ifndef __mboxstreambuf_h
#define __mboxstreambuf_h

#include <iostream>
#include <ios>

/*
  Version 1.10 (29 Apr 2002)
  changed the meaning of \r: when a CR is received, it is translated into a newline
  _within_ the current mbox. A newline means as before (ie. output buffer into mbox)

  Version 1.20 (6 Nov 2006)
  * defined class and associated data within "libwin" namespace.
  * removed pragma once directive, as this is unnecessary
  * changed unique define to conform with that used in other libraries
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libwin {

   class CMBoxStreamBuf : public std::streambuf  
{
protected:
   CString buffer;
public:
        CMBoxStreamBuf();
        virtual ~CMBoxStreamBuf();
   int underflow();
   int overflow(int nCh = EOF);
};

inline int CMBoxStreamBuf::underflow()
   {
   return EOF;
   }

inline int CMBoxStreamBuf::overflow(int nCh)
   {
   if(nCh=='\n')
      {
      if(!buffer.IsEmpty())
         AfxMessageBox(buffer);
      buffer = "";
      }
   else if(nCh=='\r')
      buffer += '\n';
   else
      buffer += char(nCh);
   return 1;
   };

}; // end namespace

#endif
