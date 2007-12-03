#ifndef __archivestreambuf_h
#define __archivestreambuf_h

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

class CArchiveStreamBuf : public std::streambuf  
{
protected:
   CArchive *ar;
public:
        CArchiveStreamBuf(CArchive* ar);
   virtual ~CArchiveStreamBuf() {};
   int underflow();
   int overflow(int nCh = EOF);
};

inline int CArchiveStreamBuf::underflow()
   {
   char c;
   return ar->Read(&c, 1) == 1 ? c : EOF;
   }

inline int CArchiveStreamBuf::overflow(int nCh)
   {
   char c = nCh;
   ar->Write(&c, 1);
   return 1;
   };

}; // end namespace

#endif
