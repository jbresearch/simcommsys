#include "stdafx.h"
#include "StringStreamBuf.h"

namespace libwin {

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CStringStreamBuf::CStringStreamBuf(CString *string)
   {
   buffer = string;
   }

CStringStreamBuf::~CStringStreamBuf()
   {
   }

} // end namespace
