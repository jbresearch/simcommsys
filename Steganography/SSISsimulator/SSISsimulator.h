#ifndef __ssissimulator_h
#define __ssissimulator_h

#include "config.h"
#include "matrix.h"
#include "stegosystem.h"
#include "image.h"

#include <string>

/*******************************************************************************
  Version 1.10 (10 Nov 2006)
  * updated to use library namespaces.
  * removed use of "using namespace std", replacing by tighter "using"
  statements as needed.

  Version 1.11 (1 Dec 2006)
  * updated to accomodate changes to stegosystem (move from libwin to libcomm
  and change of name).

  Version 1.20 (14 Dec 2006)
  * updated to use my own image class, rather than depend on the CxImage
  library. This is consonant with the move to use the FreeImage library instead
  of CxImage, and with the long-term goal of having the complete steganography
  system work in Linux as well as Windows.
  * ConvertChannel functions have been removed, as their function is now
  contained within the image class.
  * removed CWinApp parent, as well as the use of resource.h & stdafx.h

*******************************************************************************/

class CSSISsimulator : protected libcomm::stegosystem
{
// Variables
private:
   // command-line
   std::string   m_sCommand;
   // User parameters
   std::string   m_sFileInput, m_sFileOutput;
   int      m_nFormat, m_nQuality;        // Format: 0 - TIF/None, 1 - TIF/LZW, 2 - JPG
   double   m_dStrength;
   int      m_nKeySeed, m_nMessageSeed;
   // Internal variables
   int      m_nPixels;
   libbase::vector<double> m_vdMessage;

// Functions
private:
   // required by StegoSystem
   int GetImagePixels() const { return m_nPixels; };
   void DisplayProgress(const int nComplete, const int nTotal, const int nIteration, const int nTotalIterations) const {};

   // pre- and post-embedding processing (creation of data sequence, etc)
   void PreEmbedding();
   void PostEmbedding();
   // complete image process (loop over all channels)
   void ProcessImage(libimage::image& iImage);

public:
   // Constructor / Destructor
        CSSISsimulator();
        virtual ~CSSISsimulator();
   // Main interface functions
   void InterpretParameters(int argc, char *argv[]);
   int MainProcess();
};

#endif
