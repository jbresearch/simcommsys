#ifndef afx_automateembedding_h
#define afx_automateembedding_h

#ifndef __AFXWIN_H__
   #error include 'stdafx.h' before including this file for PCH
#endif

#include "Resource.h"      // main symbols
#include "PSAutomate.h"

/////////////////////////////////////////////////////////////////////////////
// SAutomateEmbeddingData
//

/*
  Data Version 1.00 (13 Oct 2003)
  initial version
*/
struct SAutomateEmbeddingData {
   // path for output files
   char     sOutput[256];
   // system options
   bool     bJpeg;
   // variables - range of embedding strengths
   double   dStrengthMin;
   double   dStrengthMax;
   double   dStrengthStep;
   // variables - range of JPEG compression quality (if requested)
   int      nJpegMin;
   int      nJpegMax;
   int      nJpegStep;
   };

/////////////////////////////////////////////////////////////////////////////
// CAutomateEmbeddingApp
// See AutomateEmbedding.cpp for the implementation of this class
//

/*
  Version 1.00 (13 Oct 2003)
  * initial version

  Version 1.10 (13 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class CAutomateEmbeddingApp : public CWinApp, public libwin::CPSAutomate
{
protected:
   SAutomateEmbeddingData* m_sData;

protected:
   // internal functions
   void DoExtract(double dStrength);

   // virtual overrides - data handling
   void ShowDialog(void);
   void InitParameters();

   // scripting support
   void WriteScriptParameters(PIActionDescriptor descriptor);
   void ReadScriptParameters(PIActionDescriptor descriptor);

public:
   CAutomateEmbeddingApp();
   virtual ~CAutomateEmbeddingApp();

   // virtual overrides - plug-in interface
   void About(void);
   void Process(void);

// Overrides
   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAutomateEmbeddingApp)
   //}}AFX_VIRTUAL

   //{{AFX_MSG(CAutomateEmbeddingApp)
      // NOTE - the ClassWizard will add and remove member functions here.
      //    DO NOT EDIT what you see in these blocks of generated code !
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif
