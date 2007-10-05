#ifndef __folderdialog_h
#define __folderdialog_h

/*
  Version 1.00 (7 Sep 2002)
  initial version - based on "Louis, Louis" MP3 player by Mark Nelson (DDJ).
  Provides a folder browse dialog box from Microsoft's library.
  Bugs: is not modal.

  Version 1.01 (21 Sep 2002)
  fixed a bug where the prompt was not being correctly used in the opened dialog.

  Version 1.02 (22 Sep 2002)
  fixed modality issue by adding parent window info to class & creator.

  Version 1.10 (6 Nov 2006)
  * defined class and associated data within "libwin" namespace.
  * removed pragma once directive, as this is unnecessary
  * changed unique define to conform with that used in other libraries
*/

namespace libwin {

class CFolderDialog  
{
public:
	CFolderDialog(const CString sPrompt, const CString sFolder, HWND hWnd=NULL);
	virtual ~CFolderDialog();

	int DoModal();
	CString GetFolder();

protected:
   static int CALLBACK BrowseCallbackProc(HWND hwnd, UINT uMsg, LPARAM lParam, LPARAM lpData);

protected:
	CString m_sPrompt;
	CString m_sFolder;
   HWND m_hWnd;
};

}; // end namespace

#endif
