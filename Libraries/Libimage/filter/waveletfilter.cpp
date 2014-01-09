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

#include "waveletfilter.h"
#include "itfunc.h"
#include <algorithm>

namespace libimage {

using std::cerr;
using libbase::trace;

using libbase::vector;
using libbase::matrix;

// helper functions

void waveletfilter::createmask(matrix<bool>& mask, const int xsize,
      const int ysize) const
   {
   // initialize mask with all elements selected
   mask.init(xsize, ysize);
   mask = true;
   // create mask-out for low-freq coefficients from the histogram
   const int xlimit = m_wWavelet.getlimit(xsize, m_nWaveletLevel);
   const int ylimit = m_wWavelet.getlimit(ysize, m_nWaveletLevel);
   matrix<bool> masklow(xlimit, ylimit);
   masklow = false;
   // copy into main mask
   mask.copyfrom(masklow);
   }

// initialization

void waveletfilter::init(const int nType, const int nPar, const int nLevel,
      const int nThreshType, const int nThreshSelector,
      const double dThreshCutoff)
   {
   m_wWavelet.init(nType, nPar);
   m_nWaveletLevel = nLevel;
   m_nThreshType = nThreshType;
   m_nThreshSelector = nThreshSelector;
   m_dThreshCutoff = dThreshCutoff;
   }

// parameter estimation (updates internal statistics)

void waveletfilter::reset()
   {
   m_vdCoefficient.clear();
   m_nSize = 0;
   }

void waveletfilter::update(const matrix<double>& in)
   {
   switch (m_nThreshSelector)
      {
      // % of coefficients
      case 0:
         {
         // do the wavelet transform of this matrix at the requested level
         matrix<double> out;
         m_wWavelet.transform(in, out, m_nWaveletLevel);
         // mask out low-frequency coefficients
         matrix<bool> mask;
         createmask(mask, in.size().rows(), in.size().cols());
         // convert masked section to a vector
         vector<double> v = out.mask(mask);
         // append values to the list
         for (int i = 0; i < v.size(); i++)
            m_vdCoefficient.push_back(fabs(v(i)));
         }
         break;
         // Hybrid+
      case 4:
         // SURE
      case 3:
         // Minimax
      case 2:
         // Visu
      case 1:
         {
         // do the wavelet transform of this matrix at level 1 only
         matrix<double> out;
         m_wWavelet.transform(in, out, m_nWaveletLevel);
         // append [hi,hi] values to the list
         for (int i = out.size().rows() >> 1; i < out.size().rows(); i++)
            for (int j = out.size().cols() >> 1; j < out.size().cols(); j++)
               m_vdCoefficient.push_back(fabs(out(i, j)));
         // update size
         m_nSize += in.size();
         }
         break;
         // unsupported type
      default:
         {
         cerr << "Unknown threshold selector (" << m_nThreshSelector << ")." << std::endl;
         }
         break;
      }
   }

void waveletfilter::estimate()
   {
   sort(m_vdCoefficient.begin(), m_vdCoefficient.end());
   switch (m_nThreshSelector)
      {
      // % of coefficients
      case 0:
         {
         m_dThreshValue = m_vdCoefficient[int(round(m_vdCoefficient.size()
               * m_dThreshCutoff))];
         }
         break;
         // Visu
      case 1:
         {
         const double dSigma = m_vdCoefficient[m_vdCoefficient.size() / 2]
               / 0.6745;
         trace << "Estimated noise std = " << dSigma << " (" << 20 * log10(
               dSigma) << "dB)" << std::endl;
         m_dThreshValue = dSigma * sqrt(2 * log(double(m_nSize)));
         }
         break;
         // Minimax
      case 2:
         {
         }
         break;
         // SURE
      case 3:
         {
         }
         break;
         // Hybrid+
      case 4:
         {
         }
         break;
         // unsupported type
      default:
         {
         cerr << "Unknown threshold selector (" << m_nThreshSelector << ")." << std::endl;
         }
         break;
      }
   trace << "Threshold = " << m_dThreshValue << std::endl;
   }

// filter process loop (only updates output matrix)

void waveletfilter::process(const matrix<double>& in, matrix<double>& out) const
   {
   // initial progress
   display_progress(0, 3);

   // do the wavelet transform of this matrix
   m_wWavelet.transform(in, out, m_nWaveletLevel);
   display_progress(1, 3);

   // mask out low-frequency coefficients
   matrix<bool> hipass;
   createmask(hipass, in.size().rows(), in.size().cols());
   // compute absolute value of coefficients
   matrix<double> outabs = out;
   outabs.apply(fabs);
   // do the truncation / shrinkage
   matrix<bool> selection;
   switch (m_nThreshType)
      {
      case 0: // hard thresholding
         selection = outabs < m_dThreshValue;
         selection &= hipass;
         out.mask(selection) = 0;
         break;
      case 1: // soft thresholding
         selection = out > m_dThreshValue;
         selection &= hipass;
         out.mask(selection) -= m_dThreshValue;
         selection = out < -m_dThreshValue;
         selection &= hipass;
         out.mask(selection) += m_dThreshValue;
         selection = outabs < m_dThreshValue;
         selection &= hipass;
         out.mask(selection) = 0;
         break;
      default: // unknown
         cerr << "Unknown thresholding type (" << m_nThreshType << ")." << std::endl;
         break;
      }
   const int count = out.mask(selection).size();
   const int n = in.size();
   trace << "Retained " << n - count << " coefficients (" << 100.0
         * (n - count) / n << "%)" << std::endl;
   display_progress(2, 3);

   // do the inverse transform
   m_wWavelet.inverse(out, out, m_nWaveletLevel);
   display_progress(3, 3);
   }


} // end namespace
