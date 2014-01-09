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

#ifndef __stegosystem_h
#define __stegosystem_h

#include "config.h"
#include "serializer_libcomm.h"

namespace libcomm {

/*!
 * \brief   Stegosystem encoder/decoder.
 * \author  Johann Briffa
 *
 * \version 1.00 (15 Nov 2002)
 * initial version - created by moving embedding/extraction routines from the embed/extract
 * filters - this makes the code for the embed/extract filters leaner, and also simplifies
 * the process of keeping them in sync. The following changes were necessary:
 * - added interleaver density as a parameter to GetRawSize, GetDataSize, and DecodeData;
 * added embed rate as a parameter to GetDataSize and DecodeData; added preset strength
 * (bool flag and double value) to NormalizeGaussian.
 * - added a function FreeErrorControl(), which releases the codec and puncture system,
 * if they were allocated
 * - added a function CodecPresent() that returns true if there is an error control system
 * loaded.
 *
 * \version 1.01 (16 Nov 2002)
 * added function that computes chi square metric, given received and transmitted signals;
 * also renamed parameters in EstimateSNR in the same fashion, to indicate what they
 * represent.
 *
 * \version 1.02 (18 Nov 2002)
 * modified chi square function to return the computed value of chisq, instead of the
 * associated probability value.
 *
 * \version 1.03 (18 Nov 2002)
 * modified chi square function so that the associated probability value is only computed
 * in the debug build - this avoids potential numerical errors in the release build, where
 * the result is not used anyway.
 *
 * \version 1.04 (19 Feb 2003)
 * modified BandwidthExpander by adding a sigspace version. This allows the expander
 * to work after modulation and not necessarily before.
 *
 * \version 1.05 (17 Jul 2006)
 * added explicit conversion to int for round() in ComputeChiSquare, to conform with the
 * change in itfunc 1.07
 *
 * \version 1.06 (7 Oct 2006)
 * - added explicit conversion to int for round() in various functions, to avoid compiler
 * warning about possible data loss in conversion to integer.
 * - changed sqrt(2) to sqrt(double(2)) in various functions, to avoid ambiguity in recent
 * VS .NET due to overloading of math functions.
 *
 * \version 1.10 (6 Nov 2006)
 * - defined class and associated data within "libwin" namespace.
 * - removed pragma once directive, as this is unnecessary
 * - changed unique define to conform with that used in other libraries
 *
 * \version 1.20 (1 Dec 2006)
 * - moved class from libwin to libcomm library and associated namespace.
 * - added vcs object.
 * - changed name from CStegoSystem to stegosystem, to better reflect library usage.
 * - although not according to library practice, the function names have been left as
 * they are (camel case).
 */

class stegosystem {
private:
   const serializer_libcomm m_serializer_libcomm;
   codec<libbase::vector>* m_pCodec;

protected:
   // Piece-wise Linear Modulator
  static double plmod(const double u);

public:
   // creation/destruction
   stegosystem();
   virtual ~stegosystem();

   // required functions in derived class
   virtual int GetImagePixels() const = 0;
   virtual void DisplayProgress(int nComplete, int nTotal, int nIteration,
         int nTotalIterations) const = 0;

   // informative functions
   int GetRawSize(double dInterleaverDensity) const;
   int GetOutputSize() const;
   int GetInputSize() const;
   int GetDataSize(double dInterleaverDensity, int nEmbedRate) const;
   int GetDataWidth() const;
   double GetCodeRate() const;
   bool CodecPresent() const
      {
      return (m_pCodec != NULL);
      }

   void LoadErrorControl(const char* sCodec);
   void FreeErrorControl();
   void LoadDataFile(const char* sPathName, libbase::vector<int>& d, int n);
   void EncodeData(const libbase::vector<int>& d, libbase::vector<int>& e);
   void DecodeData(double dInterleaverDensity, int nEmbedRate,
         const double dSNR, const libbase::vector<sigspace>& s,
         libbase::vector<int>& d);
   void DemodulateData(const libbase::vector<sigspace>& s,
         libbase::vector<int>& d);
   double EstimateSNR(const double dRate, const libbase::vector<sigspace>& rx,
         const libbase::vector<sigspace>& tx, double* dSNRreal);
   double ComputeChiSquare(const libbase::vector<sigspace>& rx,
         const libbase::vector<sigspace>& tx, int nBins, double dSNR);
   void GenerateSourceSequence(libbase::vector<int>& d, int n, int seed);
   void GenerateEmbedSequence(libbase::vector<double>& u, int seed);
   void DemodulateEmbedSequence(const libbase::vector<double>& v,
         const libbase::vector<double>& u, libbase::vector<sigspace>& s);
   void ModulateEmbedSequence(const libbase::vector<int>& d,
         const libbase::vector<double>& u, libbase::vector<double>& v);
   void ConvertToUniform(const libbase::vector<double>& g, libbase::vector<
         double>& v);
   void ConvertToGaussian(const libbase::vector<double>& v, libbase::vector<
         double>& g);
   void NormalizeGaussian(libbase::vector<double>& g, bool bPresetStrength,
         double dEmbedStrength);
   void GenerateInterleaver(libbase::vector<int>& v, int in, int out, int seed);
   void DeInterleaveMessage(const libbase::vector<int>& viIndex,
         const libbase::vector<double>& vdIn, libbase::vector<double>& vdOut);
   void InterleaveMessage(const libbase::vector<int>& viIndex,
         const libbase::vector<double>& vdIn, libbase::vector<double>& vdOut);
   void BandwidthCompressor(int nRate, const libbase::vector<sigspace>& viIn,
         libbase::vector<sigspace>& viOut);
   void BandwidthExpander(int nRate, const libbase::vector<int>& viIn,
         libbase::vector<int>& viOut);
   void BandwidthExpander(int nRate, const libbase::vector<double>& viIn,
         libbase::vector<double>& viOut);
   void BandwidthExpander(int nRate, const libbase::vector<sigspace>& viIn,
         libbase::vector<sigspace>& viOut);
};

} // end namespace

#endif
