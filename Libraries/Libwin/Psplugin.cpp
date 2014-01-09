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
#include "PSPlugIn.h"
#include "itfunc.h"

#ifdef ADOBESDK

namespace libwin {

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CPSPlugIn::CPSPlugIn(const int nDataSize, const int nDataVersion)
   {
   // setup data handling information
   m_nDataSize = nDataSize;
   m_nDataVersion = nDataVersion;

   m_pFilterRecord = NULL;
   m_pData = NULL;

   m_nTileOverlap = 0;
   }

CPSPlugIn::~CPSPlugIn()
   {
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn main function

void CPSPlugIn::Main(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult)
   {
   AFX_MANAGE_STATE(AfxGetStaticModuleState());
   try
      {
      Entry(pFilterRecord, pData);

      switch(nSelector)
         {
         case filterSelectorAbout:
            FilterAbout();
            break;
         case filterSelectorParameters:
            FilterParameters();
            break;
         case filterSelectorPrepare:
            FilterPrepare();
            break;
         case filterSelectorStart:
            FilterStart();
            break;
         case filterSelectorContinue:
            FilterContinue();
            break;
         case filterSelectorFinish:
            FilterFinish();
            break;
         }

      Exit();
      }

   catch(char* inErrorString)
      {
      OutputDebugString(inErrorString);
      char *pErrorString = (char*)pFilterRecord->errorString;
      if (pErrorString != NULL)
         {
         *pErrorString = strlen(inErrorString);
         for (int a=0; a < pErrorString[0]; a++)
            {
            *++pErrorString = *inErrorString++;
            }
         *pErrorString = '\0';
         }
      *pResult = errReportString;
      }

   catch(short inError)
      {
      *pResult = inError;
      }

   catch(...)
      {
      *pResult = -1;
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn entry and exit points

// this gets called every time through the main() loop so we stay in sync
void CPSPlugIn::Entry(FilterRecord* pFilterRecord, long* pData)
   {
   TRACE("Filter Entry\n");
   if(pFilterRecord == NULL)
      throw("pFilterRecord == NULL in CPSPlugIn::Entry");
   m_pFilterRecord = pFilterRecord;
   m_pData = pData;
   }

// this gets called every time through the main() loop so we stay in sync
void CPSPlugIn::Exit()
   {
   TRACE("Filter Exit\n");
   // I am not sure if I need this
   if((char**)*m_pData != NULL)
      m_pFilterRecord->handleProcs->unlockProc((Handle)*m_pData);
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn filter selector functions

// show the about dialog here
void CPSPlugIn::FilterAbout(void)
   {
   TRACE("Filter About\n");
   }

// this will be called the first time your plug in is ran
void CPSPlugIn::FilterParameters(void)
   {
   TRACE("Filter Parameters\n");
   // if we're called here we will be showing the dialog
   m_bShowDialog = true;
   }

// the plug in could start here with the Ctrl-F command, "Run Last Filter"
void CPSPlugIn::FilterPrepare(void)
   {
   TRACE("Filter Prepare\n");
   // if data block is not yet allocated, do it now;
   // otherwise, check that it's the right version
   if(*m_pData == NULL)
      DataAllocate();
   else
      DataCheckVersion();
   }

void CPSPlugIn::FilterStart(void)
   {
   TRACE("Filter Start\n");
   // read parameters from the scripting sub-system, if given
   ReadScriptParameters();
   // show user dialog to get parameters, as required
   if(m_bShowDialog)
      ShowDialog();
   // select the first rectangle based on the given tile suggestions
   IterationStart();
   // display initial progress indicator
   DisplayTotalProgress(0);
   // start operation timer
   m_tDuration.start();
   }

void CPSPlugIn::FilterContinue(void)
   {
   TRACE("Filter Continue\n");
   // select the next rectangle based on the given tile suggestions
   TileUpdate();
   }

void CPSPlugIn::FilterFinish(void)
   {
   TRACE("Filter Finish\n");
   // stop operation timer
   m_tDuration.stop();
   TRACE("Time taken: %s\n", std::string(m_tDuration).c_str());
   // display final progress indicator
   DisplayTotalProgress(100);
   // write scripting parameters
   WriteScriptParameters();
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn private helpers - data block memory allocation

void CPSPlugIn::DataAllocate()
   {
   TRACE("Filter Allocating Data Block\n");
   // allocate memory for parameters
   char** sHandle = m_pFilterRecord->handleProcs->newProc(m_nDataSize + sizeof(int));
   if(sHandle == NULL)
      throw("Failed to create handle in CPSPlugIn::DataAllocate");
   *m_pData = (long)sHandle;
   // lock and de-reference memory handle
   char* sData = m_pFilterRecord->handleProcs->lockProc(sHandle, true);
   if(sData == NULL)
      throw("Failed to lock handle in CPSPlugIn::DataAllocate");
   // store version info
   int *pnVersion = (int *)sData;
   *pnVersion = m_nDataVersion;
   // intialise derived class parameters
   InitPointer(sData + sizeof(int));
   InitParameters();
   }

void CPSPlugIn::DataCheckVersion()
   {
   // you better have a valid handle to get your data out of
   char* sData = m_pFilterRecord->handleProcs->lockProc((char**)*m_pData, true);
   if(sData == NULL)
      throw("Failed to lock handle in CPSPlugIn::DataCheckVersion");
   // check data block version for compatibility
   int *pnVersion = (int *)sData;
   if(*pnVersion == m_nDataVersion)
      InitPointer(sData + sizeof(int));
   else
      {
      TRACE("Data block version mismatch: %d (should be %d)\n", *pnVersion, m_nDataVersion);
      // unlock the memory handle & dispose of memory
      m_pFilterRecord->handleProcs->unlockProc((Handle)*m_pData);
      m_pFilterRecord->handleProcs->disposeProc((Handle)*m_pData);
      *m_pData = NULL;
      // allocate and set up parameter block
      DataAllocate();
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn private helpers - tile manipulation

void CPSPlugIn::TileCopy()
   {
   // set up aliases for filter record values we're using
   Rect* inRect  = &m_pFilterRecord->inRect;
   Rect* outRect = &m_pFilterRecord->outRect;

   // copy the output rectangle and plane from the input
   outRect->left = inRect->left;
   outRect->top = inRect->top;
   outRect->right = inRect->right;
   outRect->bottom = inRect->bottom;

   m_pFilterRecord->outLoPlane = m_pFilterRecord->inLoPlane;
   m_pFilterRecord->outHiPlane = m_pFilterRecord->inHiPlane;
   }

void CPSPlugIn::TileSetEmpty()
   {
   // set up aliases for filter record values we're using
   Rect* inRect  = &m_pFilterRecord->inRect;

   // define input rectangle and plane as null
   inRect->top = 0;
   inRect->left = 0;
   inRect->bottom = 0;
   inRect->right = 0;

   m_pFilterRecord->inLoPlane = 0;
   m_pFilterRecord->inHiPlane = 0;

   // copy the output rectangle and plane from the input
   TileCopy();
   }

void CPSPlugIn::TileWrite()
   {
   // set up aliases for filter record values we're using
   Rect* inRect  = &m_pFilterRecord->inRect;

   // select the input rectangle coordinates & plane
   inRect->left = m_nHorTile * (GetSuggestedTileWidth() - m_nTileOverlap);
   inRect->top = m_nVerTile * (GetSuggestedTileHeight() - m_nTileOverlap);
   inRect->right = inRect->left + GetSuggestedTileWidth();
   inRect->bottom = inRect->top + GetSuggestedTileHeight();

   if(inRect->right > GetImageWidth())
      inRect->right = GetImageWidth();
   if(inRect->bottom > GetImageHeight())
      inRect->bottom = GetImageHeight();

   m_pFilterRecord->inLoPlane = m_nPlane;
   m_pFilterRecord->inHiPlane = m_nPlane;

   // inform the user for debugging
   TRACE("Tile: %d,%d,%d of %d,%d,%d (%d,%d -> %d,%d)\n", \
      m_nPlane+1, m_nVerTile+1, m_nHorTile+1, \
      GetPlanes(), GetTilesVer(), GetTilesHor(), \
      inRect->left, inRect->top, inRect->right-1, inRect->bottom-1);

   // copy the output rectangle & plane from input
   TileCopy();
   }

void CPSPlugIn::TileUpdate()
   {
   // set up aliases for filter record values we're using
   const int nLayerPlanes = GetPlanes();
   const int nTilesVer = GetTilesVer();
   const int nTilesHor = GetTilesHor();

   // find which is the next tile we need to process and update the records
   // progress is in row-major order per tile
   if(++m_nHorTile >= nTilesHor)
      {
      m_nHorTile = 0;
      if(++m_nVerTile >= nTilesVer)
         {
         m_nVerTile = 0;
         if(++m_nPlane >= nLayerPlanes)
            {
            TileSetEmpty();
            return;
            }
         }
      }
   TileWrite();

   // see if the user cancelled
   if(m_pFilterRecord->abortProc())
      {
      TileSetEmpty();
      throw((short)userCanceledErr);
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn private helpers - descriptor suite low-level access

void CPSPlugIn::ReadScriptParameters()
   {
   if(IsDescriptorAvailable())
      {
      PIReadDescriptor token = OpenReader();
      if(token != NULL)
         {
         TRACE("Filter Reading Script Parameters\n");
         InitParameters();
         DescriptorKeyID key = NULL;
         DescriptorTypeID type = NULL;
         int32 flags = 0;
         while(GetReadDescProcs()->getKeyProc(token, &key, &type, &flags))
            ReadScriptParameter(token, key, type, flags);
         CloseReader(token);
         }
      }
   }

void CPSPlugIn::WriteScriptParameters()
   {
   if(IsDescriptorAvailable())
      {
      PIWriteDescriptor token = OpenWriter();
      if(token != NULL)
         {
         TRACE("Filter Writing Script Parameters\n");
         WriteScriptParameters(token);
         CloseWriter(token);
         }
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn protected helpers - user hooks for multi-pass operators

bool CPSPlugIn::IterationDone() const
   {
   // set up aliases for filter record values we're using
   Rect* inRect  = &m_pFilterRecord->inRect;

   // check if input rectangle is null - this indicates we're ready
   return(inRect->top == 0 && \
          inRect->left == 0 && \
          inRect->bottom == 0 && \
          inRect->right == 0);
   }

void CPSPlugIn::IterationStart()
   {
   m_nPlane = 0;
   m_nVerTile = 0;
   m_nHorTile = 0;
   TileWrite();
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn protected helpers - user progress display functions

void CPSPlugIn::DisplayTotalProgress(const int nComplete, const int nTotal, const int nIteration, const int nTotalIterations) const
   {
   // compute progress over all iterations
   const int nProgressTotal = nTotalIterations * nTotal;
   const int nProgressComplete = nIteration * nTotal + nComplete;
   m_pFilterRecord->progressProc(nProgressComplete, nProgressTotal);
   }

void CPSPlugIn::DisplayTileProgress(const int nComplete, const int nTotal, const int nIteration, const int nTotalIterations) const
   {
   // compute number of pixels done for current tile
   const int nTileTotal = GetCurrentTileWidth() * GetCurrentTileHeight();
   const int nTileDone = int(floor(nComplete * nTileTotal / double(nTotal)));
   // compute number of pixels done & total to do for whole image
   const int nPixelsTotal = GetImageWidth() * GetImageHeight() * GetPlanes();
   const int nPixelsDone = GetCurrentPlane() * GetImageWidth() * GetImageHeight() + \
                           GetCurrentCoordTop() * GetImageWidth() + \
                           GetCurrentCoordLeft() * GetCurrentTileHeight() + \
                           nTileDone;
   // display progress over all iterations
   DisplayTotalProgress(nPixelsDone, nPixelsTotal, nIteration, nTotalIterations);
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn protected helpers - pixel conversion

void CPSPlugIn::GetPixelMatrix(libbase::matrix<double>& m)
   {
   const int nRowWidth = IsWide() ? m_pFilterRecord->inRowBytes/2 : m_pFilterRecord->inRowBytes;
   const int nTileWidth = GetCurrentTileWidth();
   const int nTileHeight = GetCurrentTileHeight();
   uint8* nSmallPixel = (uint8*) m_pFilterRecord->inData;
   uint16* nLargePixel = (uint16*) m_pFilterRecord->inData;

   m.init(nTileWidth, nTileHeight);
   if(IsWide())
      for(int y=0; y<nTileHeight; y++, nLargePixel += nRowWidth)
         for(int x=0; x<nTileWidth; x++)
            m(x,y) = nLargePixel[x] / double(0x8000);
   else
      for(int y=0; y<nTileHeight; y++, nSmallPixel += nRowWidth)
         for(int x=0; x<nTileWidth; x++)
            m(x,y) = nSmallPixel[x] / double(0xff);
   }

void CPSPlugIn::SetPixelMatrix(const libbase::matrix<double>& m)
   {
   // first define some aliases
   const int nRowWidth = IsWide() ? m_pFilterRecord->outRowBytes/2 : m_pFilterRecord->outRowBytes;
   uint8* nSmallPixel = (uint8*)(m_pFilterRecord->outData);
   uint16* nLargePixel = (uint16*)(m_pFilterRecord->outData);
   // now compute the actual start/end for horizontal and vertical coordinates
   const int nTileOffset = m_nTileOverlap/2;
   const int nXStart = GetCurrentCoordLeft()==0 ? 0 : nTileOffset;
   const int nXStop = GetCurrentCoordRight()==GetImageWidth() ? GetCurrentTileWidth() : GetCurrentTileWidth()-nTileOffset;
   const int nYStart = GetCurrentCoordTop()==0 ? 0 : nTileOffset;
   const int nYStop = GetCurrentCoordBottom()==GetImageHeight() ? GetCurrentTileHeight() : GetCurrentTileHeight()-nTileOffset;

   if(IsWide())
      {
      for(int i=0; i<nYStart; i++)
         nLargePixel += nRowWidth;
      for(int y=nYStart; y<nYStop; y++, nLargePixel += nRowWidth)
         for(int x=nXStart; x<nXStop; x++)
            nLargePixel[x] = (uint16) round(m(x,y) * 0x8000);
      }
   else
      {
      for(int i=0; i<nYStart; i++)
         nSmallPixel += nRowWidth;
      for(int y=nYStart; y<nYStop; y++, nSmallPixel += nRowWidth)
         for(int x=nXStart; x<nXStop; x++)
            nSmallPixel[x] = (uint8) round(m(x,y) * 0xff);
      }
   }

double CPSPlugIn::GetPixelValue(const int x, const int y)
   {
   const int nRowWidth = IsWide() ? m_pFilterRecord->inRowBytes/2 : m_pFilterRecord->inRowBytes;
   uint8* nSmallPixel = (uint8*) m_pFilterRecord->inData;
   uint16* nLargePixel = (uint16*) m_pFilterRecord->inData;
   const int nOffset = y*nRowWidth + x;
   if(IsWide())
      return nLargePixel[nOffset] / double(0x8000);
   else
      return nSmallPixel[nOffset] / double(0xff);
   }

void CPSPlugIn::SetPixelValue(const int x, const int y, const double c)
   {
   const int nRowWidth = IsWide() ? m_pFilterRecord->outRowBytes/2 : m_pFilterRecord->outRowBytes;
   uint8* nSmallPixel = (uint8*) m_pFilterRecord->outData;
   uint16* nLargePixel = (uint16*) m_pFilterRecord->outData;
   const int nOffset = y*nRowWidth + x;
   if(IsWide())
      nLargePixel[nOffset] = (uint16) round(c * 0x8000);
   else
      nSmallPixel[nOffset] = (uint8) round(c * 0xff);
   }

/////////////////////////////////////////////////////////////////////////////
// CPSPlugIn protected helpers - scripting support

void CPSPlugIn::CloseWriter(PIWriteDescriptor token)
   {
   // delete the old descriptor key/value set
   m_pFilterRecord->handleProcs->disposeProc(GetDescParams()->descriptor);
   // close the write descriptor handle, and create new descriptor key/value set
   PIDescriptorHandle h;
   GetWriteDescProcs()->closeWriteDescriptorProc(token, &h);
   // return results to photoshop
   GetDescParams()->descriptor = h;
   GetDescParams()->recordInfo = plugInDialogOptional;
   }

void CPSPlugIn::PutString(PIWriteDescriptor token, DescriptorKeyID key, const char *data)
   {
   Str255 sTemp;
   sTemp[0] = strlen(data);
   strcpy((char *)sTemp+1, data);
   GetWriteDescProcs()->putStringProc(token, key, sTemp);
   }

void CPSPlugIn::PutInteger(PIWriteDescriptor token, DescriptorKeyID key, int data)
   {
   GetWriteDescProcs()->putIntegerProc(token, key, data);
   }

void CPSPlugIn::PutFloat(PIWriteDescriptor token, DescriptorKeyID key, double data)
   {
   GetWriteDescProcs()->putFloatProc(token, key, &data);
   }

void CPSPlugIn::PutBoolean(PIWriteDescriptor token, DescriptorKeyID key, bool data)
   {
   GetWriteDescProcs()->putBooleanProc(token, key, data);
   }

void CPSPlugIn::CloseReader(PIReadDescriptor token)
   {
   // close the read descriptor handle
   GetReadDescProcs()->closeReadDescriptorProc(token);
   // delete the old descriptor key/value set
   m_pFilterRecord->handleProcs->disposeProc(GetDescParams()->descriptor);
   // return results to photoshop
   GetDescParams()->descriptor = NULL;
   GetDescParams()->playInfo = plugInDialogDisplay;
   }

bool CPSPlugIn::GetString(PIReadDescriptor token, char *data)
   {
   Str255 temp;
   if(!GetReadDescProcs()->getStringProc(token, &temp))
      {
      strcpy(data, (char *)temp+1);
      return true;
      }
   return false;
   }

bool CPSPlugIn::GetInteger(PIReadDescriptor token, int *data)
   {
   int32 temp;
   if(!GetReadDescProcs()->getIntegerProc(token, &temp))
      {
      *data = temp;
      return true;
      }
   return false;
   }

bool CPSPlugIn::GetFloat(PIReadDescriptor token, double *data)
   {
   double temp;
   if(!GetReadDescProcs()->getFloatProc(token, &temp))
      {
      *data = temp;
      return true;
      }
   return false;
   }

bool CPSPlugIn::GetBoolean(PIReadDescriptor token, bool *data)
   {
   Boolean temp;
   if(!GetReadDescProcs()->getBooleanProc(token, &temp))
      {
      *data = temp!=0;
      return true;
      }
   return false;
   }

} // end namespace

#endif
