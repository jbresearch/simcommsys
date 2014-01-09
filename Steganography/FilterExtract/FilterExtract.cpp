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
#include "FilterExtract.h"
#include "FilterExtractDlg.h"
#include "DisplayResultsDlg.h"
#include "ScriptingKeys.h"
#include <math.h>
#include "limiter.h"
#include "fbstream.h"
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractApp

BEGIN_MESSAGE_MAP(CFilterExtractApp, CWinApp)
//{{AFX_MSG_MAP(CFilterExtractApp)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractApp construction

CFilterExtractApp::CFilterExtractApp() : CPSPlugIn(sizeof(SFilterExtractData), 135)
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractApp filter selector functions

// show the about dialog here
void CFilterExtractApp::FilterAbout(void)
   {
   CAboutDlg dlg;
   dlg.DoModal();
   }

void CFilterExtractApp::FilterStart(void)
   {
   // tile the image row by row, keeping the same tile area as suggested
   SetTileWidth(GetImageWidth());
   SetTileHeight(max(1,GetSuggestedTileHeight()*GetSuggestedTileWidth()/GetImageWidth()));
   // show dialog if necessary, set up first tile, start progress indicator & timer
   CPSPlugIn::FilterStart();
   // set up multi-pass counter
   m_nIteration = 0;

   // do processing that can be done for whole image
   m_vdMessage.init(GetImagePixels());
   }

void CFilterExtractApp::FilterContinue(void)
   {
   // set up library names
   using std::ofstream;
   using libbase::trace;
   using libbase::vector;
   using libbase::matrix;
   using libbase::weight;
   using libcomm::sigspace;
   using libimage::limiter;

   switch(m_nIteration)
      {
      case 0: {
         // update progress counter
         DisplayTileProgress(0, 100, 0, CodecPresent() ? 5 : 3);

         // convert tile to matrix
         matrix<double> m;
         GetPixelMatrix(m);

         // extract from image
         int k = GetCurrentPlane() * GetImageWidth() * GetImageHeight() + GetCurrentCoordTop() * GetImageWidth();
         for(int j=0; j<m.size().cols(); j++)
            for(int i=0; i<m.size().rows(); i++)
               m_vdMessage(k++) = m(i,j);
         } break;

      case 1: {
         // update progress counter
         DisplayTileProgress(0, 100, 0, 1);

         // convert tile to matrix
         matrix<double> m;
         GetPixelMatrix(m);

         // replace image
         int k = GetCurrentPlane() * GetImageWidth() * GetImageHeight() + GetCurrentCoordTop() * GetImageWidth();
         for(int j=0; j<m.size().cols(); j++)
            for(int i=0; i<m.size().rows(); i++)
               m(i,j) = m_vdMessage(k++);

         // clip & convert matrix to tile
         limiter<double> lim(0,1);
         lim.process(m);
         SetPixelMatrix(m);
         } break;
      }

   // select the next rectangle based on the given tile suggestions
   CPSPlugIn::FilterContinue();
   // if we have finished extracting the message
   if(m_nIteration==0 && IterationDone())
      {
      // de-interleave if necessary
      vector<int> viIndex;
      if(m_sData->bInterleave)
         {
         GenerateInterleaver(viIndex, GetImagePixels(), GetRawSize(m_sData->dInterleaverDensity), m_sData->nInterleaverSeed);
         const vector<double> vdCopy = m_vdMessage;
         DeInterleaveMessage(viIndex, vdCopy, m_vdMessage);
         }
      // save extracted data (image domain; encoded but de-interleaved)
      if(strlen(m_sData->sExtractedImage) != 0)
         {
         ofstream file(m_sData->sExtractedImage);
         for(int i=0; i<m_vdMessage.size(); i++)
            file << m_vdMessage(i) << "\n";
         }

      // rescale message to be gaussian with mean 0 and var 1
      NormalizeGaussian(m_vdMessage, m_sData->bPresetStrength, m_sData->dEmbedStrength);
      // convert to uniform
      ConvertToUniform(m_vdMessage, m_vdMessage);
      // re-create pseudo-noise sequence (as used in source)
      vector<double> u(m_vdMessage.size());
      GenerateEmbedSequence(u, m_sData->nEmbedSeed);
      // de-modulate sequence (re-build sigspace vector)
      vector<sigspace> s;
      DemodulateEmbedSequence(m_vdMessage, u, s);
      // do bandwidth compression, if necessary
      if(m_sData->nEmbedRate > 1)
         {
         const vector<sigspace> scopy = s;
         BandwidthCompressor(m_sData->nEmbedRate, scopy, s);
         }

      // load codec & puncture files as necessary
      LoadErrorControl(m_sData->sCodec, m_sData->sPuncture);
      // load/create data sequence
      vector<int> d(GetDataSize(m_sData->dInterleaverDensity, m_sData->nEmbedRate));
      switch(m_sData->nSourceType)
         {
         case 0:
            d = 0;
            break;
         case 1:
            d = (1<<GetDataWidth())-1;
            break;
         case 2:
            GenerateSourceSequence(d, GetDataWidth(), m_sData->nSourceSeed);
            break;
         case 3:
            LoadDataFile(m_sData->sSource, d, GetDataWidth());
            break;
         default:
            throw("Unsupported source type.");
            break;
         }
      // encode & puncture it, if necessary
      vector<int> d2;
      if(CodecPresent())
         {
         d2.init(GetRawSize(m_sData->dInterleaverDensity)/m_sData->nEmbedRate);
         EncodeData(d, d2);
         }
      else
         d2 = d;

      // save embedded data (image domain; encoded but de-interleaved)
      if(strlen(m_sData->sEmbeddedImage) != 0)
         {
         // do bandwidth expansion, if necessary
         vector<int> d3;
         if(m_sData->nEmbedRate > 1)
            {
            d3.init(GetRawSize(m_sData->dInterleaverDensity));
            BandwidthExpander(m_sData->nEmbedRate, d2, d3);
            }
         else
            d3 = d2;
         // modulate the sequence
         ModulateEmbedSequence(d3, u, m_vdMessage);
         // convert to gaussian
         ConvertToGaussian(m_vdMessage, m_vdMessage);
         // scale gaussian sequence if strength given
         if(m_sData->bPresetStrength)
            m_vdMessage *= pow(10.0, m_sData->dEmbedStrength/20.0);
         // save to file
         ofstream file(m_sData->sEmbeddedImage);
         for(int i=0; i<m_vdMessage.size(); i++)
            file << m_vdMessage(i) << "\n";
         }

      // then modulate it (to get the vector in signal space)
      vector<sigspace> e;
      libcomm::mpsk modem(2);
      modem.modulate(2, d2, e);
      // estimate SNR
      m_dSNRest = EstimateSNR(GetCodeRate(), s, e, &m_dSNRreal);
      // compute chi square metric
      m_dChiSquare = ComputeChiSquare(s, e, 51, m_dSNRreal);
      // decode it or demodulate it, as necessary
      vector<int> r;
      if(CodecPresent())
         DecodeData(m_sData->dInterleaverDensity, m_sData->nEmbedRate, m_dSNRreal, s, r);
      else
         DemodulateData(s, r);

      // save embedded data (encoded but de-interleaved)
      if(strlen(m_sData->sEmbedded) != 0)
         {
         // save the in-phase component only
         ofstream file(m_sData->sEmbedded);
         for(int i=0; i<e.size(); i++)
            file << e(i).i() << "\n";
         }
      // save extracted data (encoded but de-interleaved)
      if(strlen(m_sData->sExtracted) != 0)
         {
         // save the in-phase component only
         ofstream file(m_sData->sExtracted);
         for(int i=0; i<s.size(); i++)
            file << s(i).i() << "\n";
         }
      // save uniform random sequence
      if(strlen(m_sData->sUniform) != 0)
         {
         ofstream file(m_sData->sUniform);
         for(int i=0; i<u.size(); i++)
            file << u(i) << "\n";
         }
      // save decoded sequence
      if(strlen(m_sData->sDecoded) != 0)
         {
         libbase::ofbstream file(m_sData->sDecoded);
         libbase::bitfield b;
         b.resize(1);
         for(int i=0; i<r.size(); i++)
            {
            b = r(i);
            file << b;
            }
         }

      // compare decoded with embedded data sequence to compute BER, if necessary
      if(m_sData->bPrintBER)
         {
         vector<int> e = d;
         trace << "Comparing embedded data (" << e.size() << ") with decoded data (" << r.size() << ").\n";
         e ^= r;
         e.apply(weight);
         m_nCount = e.sum();
         m_nLength = r.size() * GetDataWidth();
         m_dBER = m_nCount/double(m_nLength);
         trace << "Hard errors: " << m_nCount << "/" << m_nLength << " = " << 100*m_dBER << "%\n";
         }

      // compute channel error, if requested
      if(m_sData->nFeedback > 0)
         {
         // compute in-phase error and scale from [-4,4] to [0,1]
         m_vdMessage.init(s.size());
         switch(m_sData->nFeedback)
            {
            case 1: { // Channel Error
               for(int i=0; i<s.size(); i++)
                  m_vdMessage(i) = 0.5 + (s(i).i() - e(i).i()) / 8;
               } break;
            case 2: { // Absolute Error
               for(int i=0; i<s.size(); i++)
                  m_vdMessage(i) = fabs(s(i).i() - e(i).i()) / 4;
               } break;
            case 3: { // Probability Ratio
               // set up channel
               libcomm::laplacian chan;
               chan.set_eb(1);
               chan.set_snr(m_dSNRreal);
               // compute log probability ratio
               for(int i=0; i<s.size(); i++)
                  m_vdMessage(i) = -log10(chan.pdf(e(i), s(i)) / chan.pdf(-e(i), s(i)));
               // normalize
               m_vdMessage /= 2 * max(-m_vdMessage.min(), m_vdMessage.max());
               m_vdMessage += 0.5;
               } break;
            }
         // do bandwidth expansion, if necessary
         if(m_sData->nEmbedRate > 1)
            {
            const vector<double> vdCopy = m_vdMessage;
            m_vdMessage.init(GetRawSize(m_sData->dInterleaverDensity));
            BandwidthExpander(m_sData->nEmbedRate, vdCopy, m_vdMessage);
            }
         // interleave if necessary
         if(m_sData->bInterleave)
            {
            const vector<double> vdCopy = m_vdMessage;
            InterleaveMessage(viIndex, vdCopy, m_vdMessage);
            }
         // set up for next iteration (only if there is feedback)
         m_nIteration++;
         IterationStart();
         }
      }
   }

void CFilterExtractApp::FilterFinish(void)
   {
   // stop timer & show final progress indication
   CPSPlugIn::FilterFinish();

   // display results (SNR & BER) if necessary
   if(m_sData->bPrintBER || m_sData->bPrintSNR)
      {
      CDisplayResultsDlg dlg;
      // actual BER
      if(m_sData->bPrintBER)
         {
         dlg.m_sBER.Format("%g%% (%d/%d)", 100*m_dBER, m_nCount, m_nLength);
         }
      // actual and estimated SNR
      if(m_sData->bPrintSNR)
         {
         dlg.m_sSNR.Format("%0.3f dB", m_dSNRreal);
         dlg.m_sSNRest.Format("%0.3f dB", m_dSNRest);
         }
      // actual ChiSquare
      if(m_sData->bPrintChiSquare)
         {
         //dlg.m_sChiSquare.Format("%g%%", 100*m_dChiSquare);
         dlg.m_sChiSquare.Format("%0.2f", m_dChiSquare);
         }
      // code rate
      dlg.m_sRate.Format("%g (%g:1)", GetCodeRate(), 1/GetCodeRate());
      // save results - otherwise show the dialog
      if(strlen(m_sData->sResults) != 0)
         {
         std::ofstream file(m_sData->sResults, std::ios::out | std::ios::app);
         if(m_sData->bPrintBER)
            file << "\t" << m_dBER;
         if(m_sData->bPrintSNR)
            file << "\t" << m_dSNRreal;
         if(m_sData->bPrintEstimate)
            file << "\t" << m_dSNRest;
         if(m_sData->bPrintChiSquare)
            file << "\t" << m_dChiSquare;
         file << "\n";
         }
      else
         dlg.DoModal();
      }

   // clean up memory usage
   m_vdMessage.init(0);
   FreeErrorControl();
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractApp helper functions

void CFilterExtractApp::ShowDialog(void)
   {
   CFilterExtractDlg   dlg;

   dlg.m_pPSPlugIn = this;

   // embedding system
   dlg.m_nEmbedSeed = m_sData->nEmbedSeed;
   dlg.m_nEmbedRate = m_sData->nEmbedRate;
   dlg.m_dEmbedStrength = m_sData->dEmbedStrength;
   dlg.m_bPresetStrength = m_sData->bPresetStrength;
   // channel interleaver
   dlg.m_bInterleave = m_sData->bInterleave;
   dlg.m_nInterleaverSeed = m_sData->nInterleaverSeed;
   dlg.m_dInterleaverDensity = m_sData->dInterleaverDensity;
   // source data
   dlg.m_nSourceType = m_sData->nSourceType;
   dlg.m_nSourceSeed = m_sData->nSourceSeed;
   dlg.m_sSource = m_sData->sSource;
   // codec and puncture pattern
   dlg.m_sCodec = m_sData->sCodec;
   dlg.m_sPuncture = m_sData->sPuncture;
   // results storage
   dlg.m_sResults = m_sData->sResults;
   dlg.m_sEmbeddedImage = m_sData->sEmbeddedImage;
   dlg.m_sExtractedImage = m_sData->sExtractedImage;
   dlg.m_sEmbedded = m_sData->sEmbedded;
   dlg.m_sExtracted = m_sData->sExtracted;
   dlg.m_sUniform = m_sData->sUniform;
   dlg.m_sDecoded = m_sData->sDecoded;
   // channel parameter computation
   dlg.m_bPrintBER = m_sData->bPrintBER;
   dlg.m_bPrintSNR = m_sData->bPrintSNR;
   dlg.m_bPrintEstimate = m_sData->bPrintEstimate;
   dlg.m_bPrintChiSquare = m_sData->bPrintChiSquare;
   dlg.m_nFeedback = m_sData->nFeedback;

   int err = dlg.DoModal();
   if(err != IDOK)
      throw((short)userCanceledErr);

   // embedding system
   m_sData->nEmbedSeed = dlg.m_nEmbedSeed;
   m_sData->nEmbedRate = dlg.m_nEmbedRate;
   m_sData->dEmbedStrength = dlg.m_dEmbedStrength;
   m_sData->bPresetStrength = dlg.m_bPresetStrength != 0;
   // channel interleaver
   m_sData->bInterleave = dlg.m_bInterleave != 0;
   m_sData->nInterleaverSeed = dlg.m_nInterleaverSeed;
   m_sData->dInterleaverDensity = dlg.m_dInterleaverDensity;
   // source data
   m_sData->nSourceType = dlg.m_nSourceType;
   m_sData->nSourceSeed = dlg.m_nSourceSeed;
   strcpy(m_sData->sSource, dlg.m_sSource);
   // codec and puncture pattern
   strcpy(m_sData->sCodec, dlg.m_sCodec);
   strcpy(m_sData->sPuncture, dlg.m_sPuncture);
   // results storage
   strcpy(m_sData->sResults, dlg.m_sResults);
   strcpy(m_sData->sEmbeddedImage, dlg.m_sEmbeddedImage);
   strcpy(m_sData->sExtractedImage, dlg.m_sExtractedImage);
   strcpy(m_sData->sEmbedded, dlg.m_sEmbedded);
   strcpy(m_sData->sExtracted, dlg.m_sExtracted);
   strcpy(m_sData->sUniform, dlg.m_sUniform);
   strcpy(m_sData->sDecoded, dlg.m_sDecoded);
   // channel parameter computation
   m_sData->bPrintBER = dlg.m_bPrintBER != 0;
   m_sData->bPrintSNR = dlg.m_bPrintSNR != 0;
   m_sData->bPrintEstimate = dlg.m_bPrintEstimate != 0;
   m_sData->bPrintChiSquare = dlg.m_bPrintChiSquare != 0;
   m_sData->nFeedback = dlg.m_nFeedback;
   SetShowDialog(false);
   }

void CFilterExtractApp::InitPointer(char* sData)
   {
   m_sData = (SFilterExtractData *) sData;
   }

void CFilterExtractApp::InitParameters()
   {
   // embedding system
   m_sData->nEmbedSeed = 0;
   m_sData->nEmbedRate = 1;
   m_sData->dEmbedStrength = -30;
   m_sData->bPresetStrength = false;
   // channel interleaver
   m_sData->bInterleave = true;
   m_sData->nInterleaverSeed = 0;
   m_sData->dInterleaverDensity = 1.0;
   // source data
   m_sData->nSourceType = 0;
   m_sData->nSourceSeed = 1;
   strcpy(m_sData->sSource, "");
   // codec and puncture pattern
   strcpy(m_sData->sCodec, "");
   strcpy(m_sData->sPuncture, "");
   // results storage
   strcpy(m_sData->sResults, "");
   strcpy(m_sData->sEmbeddedImage, "");
   strcpy(m_sData->sExtractedImage, "");
   strcpy(m_sData->sEmbedded, "");
   strcpy(m_sData->sExtracted, "");
   strcpy(m_sData->sUniform, "");
   strcpy(m_sData->sDecoded, "");
   // channel parameter computation
   m_sData->bPrintBER = true;
   m_sData->bPrintSNR = true;
   m_sData->bPrintEstimate = true;
   m_sData->bPrintChiSquare = true;
   m_sData->nFeedback = 0;
   }

/////////////////////////////////////////////////////////////////////////////
// CFilterExtractApp scripting support

void CFilterExtractApp::ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags)
   {
   switch (key)
      {
      // embedding system
      case keyEmbedSeed:
         GetInteger(token, &m_sData->nEmbedSeed);
         break;
      case keyEmbedRate:
         GetInteger(token, &m_sData->nEmbedRate);
         break;
      case keyEmbedStrength:
         GetFloat(token, &m_sData->dEmbedStrength);
         break;
      case keyPresetStrength:
         GetBoolean(token, &m_sData->bPresetStrength);
         break;
         // channel interleaver
      case keyInterleave:
         GetBoolean(token, &m_sData->bInterleave);
         break;
      case keyInterleaverSeed:
         GetInteger(token, &m_sData->nInterleaverSeed);
         break;
      case keyInterleaverDensity:
         GetFloat(token, &m_sData->dInterleaverDensity);
         break;
         // source data
      case keySourceType:
         GetInteger(token, &m_sData->nSourceType);
         break;
      case keySourceSeed:
         GetInteger(token, &m_sData->nSourceSeed);
         break;
      case keySource:
         GetString(token, m_sData->sSource);
         break;
         // codec and puncture pattern
      case keyCodec:
         GetString(token, m_sData->sCodec);
         break;
      case keyPuncture:
         GetString(token, m_sData->sPuncture);
         break;
      // results storage
      case keyResults:
         GetString(token, m_sData->sResults);
         break;
      case keyEmbeddedImage:
         GetString(token, m_sData->sEmbeddedImage);
         break;
      case keyExtractedImage:
         GetString(token, m_sData->sExtractedImage);
         break;
      case keyEmbedded:
         GetString(token, m_sData->sEmbedded);
         break;
      case keyExtracted:
         GetString(token, m_sData->sExtracted);
         break;
      case keyUniform:
         GetString(token, m_sData->sUniform);
         break;
      case keyDecoded:
         GetString(token, m_sData->sDecoded);
         break;
      // channel parameter computation
      case keyPrintBER:
         GetBoolean(token, &m_sData->bPrintBER);
         break;
      case keyPrintSNR:
         GetBoolean(token, &m_sData->bPrintSNR);
         break;
      case keyPrintEstimate:
         GetBoolean(token, &m_sData->bPrintEstimate);
         break;
      case keyPrintChiSquare:
         GetBoolean(token, &m_sData->bPrintChiSquare);
         break;
      case keyFeedback:
         GetInteger(token, &m_sData->nFeedback);
         break;
      // unknown
      default:
         libbase::trace << "key Unknown!\n";
         break;
      }
   }

void CFilterExtractApp::WriteScriptParameters(PIWriteDescriptor token)
   {
   // embedding system
   PutInteger(token, keyEmbedSeed, m_sData->nEmbedSeed);
   PutInteger(token, keyEmbedRate, m_sData->nEmbedRate);
   PutFloat(token, keyEmbedStrength, m_sData->dEmbedStrength);
   if(m_sData->bPresetStrength)
      PutBoolean(token, keyPresetStrength, m_sData->bPresetStrength);
   // channel interleaver
   PutBoolean(token, keyInterleave, m_sData->bInterleave);
   if(m_sData->bInterleave)
      {
      PutInteger(token, keyInterleaverSeed, m_sData->nInterleaverSeed);
      PutFloat(token, keyInterleaverDensity, m_sData->dInterleaverDensity);
      }
   // source data
   PutInteger(token, keySourceType, m_sData->nSourceType);
   PutInteger(token, keySourceSeed, m_sData->nSourceSeed);
   if(strlen(m_sData->sSource) > 0)
      PutString(token, keySource, m_sData->sSource);
   // codec and puncture pattern
   if(strlen(m_sData->sCodec) > 0)
      PutString(token, keyCodec, m_sData->sCodec);
   if(strlen(m_sData->sPuncture) > 0)
      PutString(token, keyPuncture, m_sData->sPuncture);
   // results storage
   if(strlen(m_sData->sResults) > 0)
      PutString(token, keyResults, m_sData->sResults);
   if(strlen(m_sData->sEmbeddedImage) > 0)
      PutString(token, keyEmbeddedImage, m_sData->sEmbeddedImage);
   if(strlen(m_sData->sExtractedImage) > 0)
      PutString(token, keyExtractedImage, m_sData->sExtractedImage);
   if(strlen(m_sData->sEmbedded) > 0)
      PutString(token, keyEmbedded, m_sData->sEmbedded);
   if(strlen(m_sData->sExtracted) > 0)
      PutString(token, keyExtracted, m_sData->sExtracted);
   if(strlen(m_sData->sUniform) > 0)
      PutString(token, keyUniform, m_sData->sUniform);
   if(strlen(m_sData->sDecoded) > 0)
      PutString(token, keyDecoded, m_sData->sDecoded);
   // channel parameter computation
   PutBoolean(token, keyPrintBER, m_sData->bPrintBER);
   PutBoolean(token, keyPrintSNR, m_sData->bPrintSNR);
   PutBoolean(token, keyPrintEstimate, m_sData->bPrintEstimate);
   PutBoolean(token, keyPrintChiSquare, m_sData->bPrintChiSquare);
   PutInteger(token, keyFeedback, m_sData->nFeedback);
   }

/////////////////////////////////////////////////////////////////////////////
// The one and only CFilterExtractApp object

CFilterExtractApp theApp;

DLLExport SPAPI void PluginMain(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   theApp.Main(nSelector, pFilterRecord, pData, pResult);
   }
