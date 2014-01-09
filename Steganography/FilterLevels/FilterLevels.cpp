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
#include "FilterLevels.h"
#include "FilterLevelsDlg.h"
#include "DisplayResultsDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "itfunc.h"
#include "limiter.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// Constants

const int CFilterLevelsApp::m_nLevels = 256;

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsApp

BEGIN_MESSAGE_MAP(CFilterLevelsApp, CWinApp)
//{{AFX_MSG_MAP(CFilterLevelsApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsApp construction

CFilterLevelsApp::CFilterLevelsApp() : CPSPlugIn(sizeof(SFilterLevelsData), 100)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsApp filter selector functions

// show the about dialog here
void CFilterLevelsApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterLevelsApp::FilterStart(void)
   {
   // FilterStart will get user parameters if necessary & select the first tile
   CPSPlugIn::FilterStart();
   // set up multi-pass counter
   m_nIteration = 0;

   // do processing that can be done for whole image:
   // set result vector/matrix sizes and initialize to zero
   m_viHistogram.init(m_nLevels);
   m_viHistogram = 0;
   m_miNeighbors.init(9,m_nLevels-1);
   m_miNeighbors = 0;
   }

void CFilterLevelsApp::FilterContinue(void)
   {
   // set up library names
   using libbase::trace;
   using libbase::vector;
   using libbase::matrix;
   using libbase::round;
   using libimage::limiter;

   switch(m_nIteration)
      {
      // obtain image histogram
      case 0: {
         // update progress counter
         DisplayTileProgress(0, 100, 0, 3);

         // convert tile to matrix
         matrix<double> m;
         GetPixelMatrix(m);

         // for each pixel, update the relevant histogram entry
         for(int j=0; j<m.size().cols(); j++)
            for(int i=0; i<m.size().rows(); i++)
               {
               int k = int(round(m(i,j) * (m_nLevels-1)));
               m_viHistogram(k)++;
               }
         } break;

      // obtain neigborhood histograms for each threshold (ignoring 1-pixel border)
      case 1: {
         // temporary variables
         vector<int> vi;

         // update progress counter
         DisplayTileProgress(0, 100, 1, 3);

         // convert tile to matrix
         matrix<double> m;
         GetPixelMatrix(m);

         // initialize thresholded matrix
         matrix<bool> mt;
         mt.init(m);

         // for each threshold
         for(int k=0; k<m_nLevels-1; k++)
            if(m_viHistogram(k) > 0)   // if occupied, perform analysis
               {
               // obtain thresholded image (where black=1, white=0)
               mt = (m <= k/double(m_nLevels-1));
               // for each pixel, if set, determine the number of neighbors
               for(int i=1; i<mt.size().rows()-1; i++)
                  for(int j=1; j<mt.size().cols()-1; j++)
                     if(mt(i,j)) // current pixel is black
                        {
                        int sum = -1;  // to discount current pixel
                        for(int ii=-1; ii<=1; ii++)
                           for(int jj=-1; jj<=1; jj++)
                              if(mt(i+ii, j+jj))
                                 sum++;
                        m_miNeighbors(sum,k)++;
                        }
               }/*
            else if(k>0)  // if unoccupied, copy from last (except for first time)
               {
               m_miNeighbors.extractcol(vi,k-1);
               m_miNeighbors.insertcol(vi,k);
               }*/
         } break;

      // adjust image levels as needed
      case 2: {
         // update progress counter
         DisplayTileProgress(0, 100, 2, 3);

         // convert tile to matrix
         matrix<double> m;
         GetPixelMatrix(m);

         // scale image values
         m = (m - m_dBlack) / (m_dWhite - m_dBlack);

         // clip & convert matrix to tile
         limiter<double> lim(0,1);
         lim.process(m);
         SetPixelMatrix(m);
         } break;
      }

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   // if we have gone over all tiles
   if(IterationDone())
      {
      // prepare for next iteration as necessary
      switch(++m_nIteration)
         {
         // obtain neigborhood histograms for each threshold
         case 1: {
            // DEBUG: output histogram matrix
            trace << "m_viHistogram:\n" << m_viHistogram << "\n";

            // set up overlap for future tile updates
            // overlap is of 2 pixels so that we can always ignore the 1-pixel border
            SetTileOverlap(2);
            // set up for next iteration
            IterationStart();
            } break;

         // adjust image levels as needed
         case 2: {
            // temporary variables
            int i,j;
            vector<int> viWeightHist(9);
            vector<double> vdNormalized(9), vdCumulative(9);
            vdCumulative = 0;

            // initialize internal variables:
            // vector with average weight for each threshold
            vector<double> vdAvgWeight(m_nLevels-1);
            vdAvgWeight = 0;
            // vector with modal weight for each threshold
            //vector<int> viModWeight(m_nLevels-1);
            //viModWeight = 0;
            // vector with median weight for each threshold
            vector<int> viMedWeight(m_nLevels-1);
            viMedWeight = 0;

            // for each threshold:
            // 1) compute the number of black pixels at each threshold
            // 2) hence obtain the normalized neighborhood weight distribution
            // 3) compute average/mode/median weight
            for(i=0; i<m_nLevels-1; i++)
               {
               // extract neighborhood weight histogram for current threshold
               m_miNeighbors.extractcol(viWeightHist,i);
               // determine the number of black pixels
               int sum = viWeightHist.sum();
               // if there are none, skip to the next threshold
               if(sum == 0)
                  continue;
               // compute the normalized neighborhood weight distribution
               for(j=0; j<9; j++)
                  vdNormalized(j) = viWeightHist(j) / double(sum);
               // compute the cumulative neighborhood weight distribution
               vdCumulative += vdNormalized;
               // compute average weight for each threshold
               for(j=1; j<9; j++)   // note: skipping zero because it does not contribute to average
                  vdAvgWeight(i) += j * vdNormalized(j);
               // compute modal weight for each threshold
               //vdNormalized.max(viModWeight(i));
               // compute median weight for each threshold
               vdNormalized = vdCumulative - 0.5;
               vdNormalized.apply(fabs);
               vdNormalized.min(viMedWeight(i));
               }
            trace << "vdAvgWeight:\n" << vdAvgWeight << "\n";
            //trace << "viModWeight:\n" << viModWeight << "\n";
            trace << "viMedWeight:\n" << viMedWeight << "\n";

            // determine ideal black point
            // (where the mode first reaches its maximum value from the black side)
            // note: original algorithm used median instead of mode
            int iBlack;
            viMedWeight.max(iBlack,true);
            m_dBlack = iBlack/double(m_nLevels-1);

            // determine ideal white point
            // (where the average first reaches its maximum value from the white side)
            int iWhite;
            vdAvgWeight.max(iWhite,false);
            m_dWhite = iWhite/double(m_nLevels-1);

            // set up overlap to zero again
            SetTileOverlap(0);
            // set up for next iteration
            IterationStart();
            } break;
         }
      }
   }

void CFilterLevelsApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();

   using libbase::round;
   CDisplayResultsDlg dlg;
   dlg.m_sBlack.Format("%0.4f (%d)", m_dBlack, round(m_dBlack * 255));
   dlg.m_sWhite.Format("%0.4f (%d)", m_dWhite, round(m_dWhite * 255));
   dlg.DoModal();

   // clean up memory usage
   m_viHistogram.init(0);
   m_miNeighbors.init(0,0);
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsApp helper functions

void CFilterLevelsApp::ShowDialog(void)
   {
   }

void CFilterLevelsApp::InitPointer(char* sData)
   {
   m_sData = (SFilterLevelsData *) sData;
   }

void CFilterLevelsApp::InitParameters()
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterLevelsApp scripting support

void CFilterLevelsApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   }

void CFilterLevelsApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterLevelsApp object

CFilterLevelsApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
