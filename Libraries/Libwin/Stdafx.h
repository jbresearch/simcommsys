// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#if !defined(AFX_STDAFX_H__BE2799D9_4B73_4F77_AFC9_9FFCAC17D8C5__INCLUDED_)
#define AFX_STDAFX_H__BE2799D9_4B73_4F77_AFC9_9FFCAC17D8C5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define VC_EXTRALEAN		// Exclude rarely-used stuff from Windows headers

#if _MSC_VER >= 1400
#	define _CRT_SECURE_NO_DEPRECATE 1
#endif

#include <afx.h>
#include <afxwin.h>     // MFC core and standard components
#include <afxmt.h>
#include <afxext.h>     // MFC extensions
#include <afxdtctl.h>	// MFC support for Internet Explorer 4 Common Controls
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>		// MFC support for Windows Common Controls
#endif // _AFX_NO_AFXCMN_SUPPORT

// additional headers used by this library
#define USING_MFC
#include <afxsock.h>

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_STDAFX_H__BE2799D9_4B73_4F77_AFC9_9FFCAC17D8C5__INCLUDED_)
