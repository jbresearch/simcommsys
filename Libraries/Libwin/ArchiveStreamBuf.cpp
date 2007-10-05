#include "stdafx.h"
#include "ArchiveStreamBuf.h"

namespace libwin {

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CArchiveStreamBuf::CArchiveStreamBuf(CArchive* ar)
   {
   CArchiveStreamBuf::ar = ar;
   }

}; // end namespace
