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

#include "SSISsimulator.h"

#include "itfunc.h"
#include "limiter.h"
#include "image.h"

#include <iostream>
#include <string.h>

using std::cerr;
using libbase::trace;

/////////////////////////////////////////////////////////////////////////////
// The one and only application object

CSSISsimulator theApp;

int main(int argc, char* argv[])
   {
   theApp.InterpretParameters(argc, argv);
   return theApp.MainProcess();
   }

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSSISsimulator::CSSISsimulator()
   {
   // Initialize defaults
   m_sFileInput = "";
   m_sFileOutput = "";
   m_nFormat = 0;
   m_nQuality = 0;
   m_dStrength = -30;
   m_nKeySeed = 0;
   m_nMessageSeed = 0;
   }

CSSISsimulator::~CSSISsimulator()
   {
   }

//////////////////////////////////////////////////////////////////////
// Main interface functions
//////////////////////////////////////////////////////////////////////

void CSSISsimulator::InterpretParameters(int argc, char *argv[])
   {
   bool error = false;
   m_sCommand = argv[0];
   // Interpret parameters in sequence
   for(int i=1; i<argc; i++)
      {
      // handle no-argument parameters first
      if(i >= argc-1)
         {
         error = true;
         continue;
         }
      // now handle two-argument parameters
      if(strcmp(argv[i],"-i") == 0)
         m_sFileInput = argv[++i];
      else if(strcmp(argv[i],"-o") == 0)
         m_sFileOutput = argv[++i];
      else if(strcmp(argv[i],"-s") == 0)
         m_dStrength = atof(argv[++i]);
      else if(strcmp(argv[i],"-k") == 0)
         m_nKeySeed = atoi(argv[++i]);
      else if(strcmp(argv[i],"-m") == 0)
         m_nMessageSeed = atoi(argv[++i]);
      else if(strcmp(argv[i],"-t") == 0)
         {
         if(strcmp(argv[++i],"none") == 0)
            m_nFormat = 0;
         else if(strcmp(argv[i],"lzw") == 0)
            m_nFormat = 1;
         else
            error = true;
         }
      else if(strcmp(argv[i],"-j") == 0)
         {
         m_nFormat = 2;
         m_nQuality = atoi(argv[++i]);
         }
      else
         error = true;
      }
   // Check for validity / completeness
   if(error || m_sFileInput.empty() || m_sFileOutput.empty())
      {
      cerr << "SSISsimulator, (c) Johann A. Briffa, 2005-2006\n";
      cerr << "  Version 1.20\n";
      cerr << "  Do not distribute. Please contact the author for license details.\n";
      cerr << "\nUsage: " << m_sCommand << " -i <infile> -o <outfile> [-t <none|lzw>|-j <q>] [-s <strength>] [-k <modulation key/seed>] [-m <message seed>]\n";
      cerr << "\nNotes:\n";
      cerr << "   * Parameter order is unimportant.\n";
      cerr << "   * -t output in TIF format; compression is none or LZW.\n";
      cerr << "   * -j output in JPG format, where Q is the quality between 0-100.\n";
      cerr << "\nDefaults:\n";
      cerr << "   * Format: TIF/None\n";
      cerr << "   * Strength: -30dB\n";
      exit(1);
      }
   }

int CSSISsimulator::MainProcess()
   {
   libimage::image iImage;
   // load input image
   std::ifstream fInput(m_sFileInput.c_str());
   fInput >> iImage;
   fInput.close();
   // do the processing, depending on color format
   m_nPixels = iImage.width() * iImage.height() * iImage.channels();
   PreEmbedding();
   ProcessImage(iImage);
   PostEmbedding();
   // save output image
   switch(m_nFormat)
      {
      case 0:  // TIF/None
         trace << "Saving in TIF/None format.\n";
         iImage.set_format(libimage::image::tiff);
         iImage.set_compression(libimage::image::none);
         break;
      case 1:  // TIF/LZW
         trace << "Saving in TIF/LZW format.\n";
         iImage.set_format(libimage::image::tiff);
         iImage.set_compression(libimage::image::lzw);
         break;
      case 2:  // JPG
         trace << "Saving in JPG format (Q=" << m_nQuality << ").\n";
         iImage.set_format(libimage::image::jpeg);
         iImage.set_quality(m_nQuality);
         break;
      default:
         cerr << "Output format (type=" << m_nFormat << ") not supported.\n";
         return 1;
      }
   // save output image
   std::ofstream fOutput(m_sFileOutput.c_str());
   fOutput << iImage;
   fOutput.close();
   return 0;
   }

//////////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////////

void CSSISsimulator::PreEmbedding()
   {
   // create data sequence
   libbase::vector<int> d(GetDataSize(1,1));
   GenerateSourceSequence(d, GetDataWidth(), m_nMessageSeed);
   trace << "Created Source (data) Sequence: length = " << d.size() << ", seed = " << m_nMessageSeed << "\n";
   // create pseudo-noise sequence
   m_vdMessage.init(d.size());
   GenerateEmbedSequence(m_vdMessage, m_nKeySeed);
   trace << "Created Embedding (pseudo-noise) Sequence: length = " << m_vdMessage.size() << ", seed = " << m_nKeySeed << "\n";
   // modulate the sequence
   ModulateEmbedSequence(d, m_vdMessage, m_vdMessage);
   //trace << "Uniform message mean = " << m_vdMessage.mean() << ", min = " << m_vdMessage.min() << ", max = " << m_vdMessage.max() << "\n";
   // convert to gaussian
   ConvertToGaussian(m_vdMessage, m_vdMessage);
   trace << "Gaussian message mean = " << m_vdMessage.mean() << ", std = " << m_vdMessage.sigma() << "\n";
   // scale gaussian sequence
   m_vdMessage *= pow(10.0, m_dStrength/20.0);
   trace << "Scaled Gaussian message mean = " << m_vdMessage.mean() << ", std = " << m_vdMessage.sigma() << "\n";
   }

void CSSISsimulator::PostEmbedding()
   {
   }

void CSSISsimulator::ProcessImage(libimage::image& iImage)
   {
   int k=0;
   for(int c=0; c<iImage.channels(); c++)
      {
      // convert channel to a matrix
      libbase::matrix<double> mChannel = iImage.getchannel(c);
      // embed in image
      for(int i=0; i<mChannel.size().rows(); i++)
         for(int j=0; j<mChannel.size().cols(); j++)
            mChannel(i,j) += m_vdMessage(k++);
      // clip
      libimage::limiter<double> lim(0,1);
      lim.process(mChannel);
      // convert back from matrix
      iImage.setchannel(c, mChannel);
      }
   }
