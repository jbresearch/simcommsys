/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "stdafx.h"
#include "FolderDialog.h"
#include "config.h"

namespace libwin {

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CFolderDialog::CFolderDialog(const CString sPrompt, const CString sFolder, HWND hWnd)
   {
   if(sPrompt.IsEmpty())
      m_sPrompt = "Select folder:";
   else
      m_sPrompt = sPrompt;
   m_sFolder = sFolder;
   m_hWnd = hWnd;
   }

CFolderDialog::~CFolderDialog()
   {
   }

//////////////////////////////////////////////////////////////////////
// Utilities
//////////////////////////////////////////////////////////////////////

int CALLBACK CFolderDialog::BrowseCallbackProc(HWND hwnd, UINT uMsg, LPARAM lParam, LPARAM lpData)
   {
   if(uMsg == BFFM_INITIALIZED)
      {
      const char *path = (char *) lpData;
      ::SendMessage( hwnd, BFFM_SETSELECTION, TRUE, (LPARAM) path );
      }
   return 0;
   }

//////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////

int CFolderDialog::DoModal()
   {
   char sDisplayName[_MAX_PATH];
   char sPath[_MAX_PATH];
   strncpy(sPath, m_sFolder, _MAX_PATH);
   BROWSEINFO browseInfo = {
      m_hWnd,              // dialog window owner
      NULL,                // browse from root
      sDisplayName,        // return space for name of folder
      m_sPrompt,
      BIF_RETURNONLYFSDIRS, // + 0x40 for new style
      BrowseCallbackProc,  // callback proc
      (LONG_PTR) sPath,    // parameter passed to callback proc
      0                    // variable to receive image
      };
   ITEMIDLIST *itemList = SHBrowseForFolder(&browseInfo);
   if(itemList)
      {
      if(SHGetPathFromIDList(itemList, sPath))
         {
         m_sFolder = sPath;
         return IDOK;
         }
      }
   return IDCANCEL;
   }

CString CFolderDialog::GetFolder()
   {
   return m_sFolder;
   }

} // end namespace
