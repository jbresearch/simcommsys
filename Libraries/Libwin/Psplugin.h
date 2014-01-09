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

#ifndef __psplugin_h
#define __psplugin_h

#include "RoutedIO.h"
#include "matrix.h"
#include "timer.h"
#include <math.h>

#ifdef ADOBESDK

#include "PIFilter.h"
#include "PIUIHooksSuite.h"

// Create a definition for exported functions
#ifndef DLLExport
#   define DLLExport extern "C" __declspec(dllexport)
#endif

/*
   \version 1.10 (27 Nov 2001)
  added informative functions (image size/planes, tile size, number of tiles, etc)

   \version 1.20 (28 Nov 2001)
  made tile manipulation functions private. Derived classes should use the pre-defined
  plug-in interface routines to get the job done.
  Also re-directed 'cerr' for all PSPlugIn-derived programs to MessageBox.
  Also moved progress indicator to TileWrite, so that the first call has a progress of
  zero - it seems that the first call is only really used to set up the indicator limits
  and does not really show up; progress is now also updated so that we count all tiles
  processed, not just the number of tile rows - this makes progress indicator work on a
  finer resolution.
  Also made most variables private. Derived classes should use the provided interfaces.
  Also added version specification for data block (if the allocated block does not match
  the required version, we re-allocate). This allows changes to the data block specification
  without reloading Photoshop.

   \version 1.30 (30 Nov 2001)
  added tile & total progress display functions.

   \version 1.31 (2 Dec 2001)
  added function to get ShowDialog flag - this is to allow user to call ShowDialog himself
  in FilterStart, allowing him to set up tiling information based on user info.

   \version 1.32 (3 Dec 2001)
  made pFilterRecord and pData as private members again - access to Photoshop info should
  be done through PSPlugIn functions only; Dialog classes can be passed a pointer to the
  PSPlugIn class rather than pFilterRecord and pData. Also added functions to set tile
  width / height, and made image & tile info functions public.

   \version 1.40 (5 Apr 2002)
  added protected functions to allow user to restart processing tile by tile. The first
  function IterationDone() returns true if all tiles have been processed in any given
  iteration (this should be called in FilterContinue only *after* the base FilterContinue
  has been called). The second function IterationStart() will set the next-tile to the
  first one, so that processing can start all over again. It is up to the user to call
  this in FilterContinue when it determines that the current iteration is done (by using
  IterationDone). It is also the user's responsibility to keep track of which iteration
  it is, and what needs to be done in this iteration.
  To make the user-interface neater, also modified the DisplayProgress functions to take
  two additional parameters - the current iteration (starting from zero) and the total
  number of iterations (which default to 0 and 1 respectively, so they can be safely
  ignored by single-pass operators). Actually these need not really be the true number
  of iterations, but only consider those iterations that are lengthy enough to require
  a progress operator.
  Also renamed TileReset to TileSetEmpty to more clearly indicate its function (this is
  a private member so it's safe to change).
  Also uncommented the informational output on current tile etc, and changed it to a
  TRACE function, so that progress can be monitored in debug builds.

   \version 1.41 (6 Apr 2002)
  removed automatic progress display from TileUpdate, since this would not work well
  in multi-pass filters. The user is requested to call one of the ProgressDisplay
  functions himself while processing the current tile. If only updated once, this
  should be done at the beginning of the tile processing, not at the end.

   \version 1.42 (7 Apr 2002)
  modified DisplayTileProgress to utilise DisplayTotalProgress for output (in order to
  centralize the code to compute the progress over several iterations). Also added
  protected functions that return current-tile coordinates for use in PSPlugIn or
  derived classes (may be used in filters that are space-variant); in the process,
  I also renamed the GetRealTileWidth and Height functions to GetCurrent... and moved
  them to protected access instead of public. They were not used except within this
  class so the move was painless. Finally, modified TileProgress to compute progress
  based on the number of pixels done/remaining, not number of tiles. Since not all
  tiles are of equal size (those at the edges are usually smaller), this change makes
  the progress indicator smoother. Also added default progress displays in FilterStart
  and FilterFinish, since these are safe and make the user interface neater (it ensures
  that the progress is reset initially and that it passes through 100% as soon as the
  filter is done). Note that this change allows the derived classes' tile progress
  display to be performed at any time during the cycle (ie removed the need for it to
  be at the start).

   \version 1.43 (7 Apr 2002)
  removed the member variable that keeps count of the number of calls to the plug-in.
  This was only used earlier for debugging purposes, to determine the calling convention
  for PhotoShop and particularly to determine when the plugin itself was reloaded.

   \version 1.44 (12 Apr 2002)
  updated tracing information (during TileWrite) so that the tile pixel ranges are
  both inclusive, and that current tile counts start from 1 rather than 0. Also, added
  trace information in most filter entry/exit points to help keep track of what is
  happening.

   \version 1.45 (12 Apr 2002)
  fixed a bug in FilterFinish, which was causing PhotoShop to complain about an error
  in the program; I was passing the timer duration as a string object (rather than a
  C-style string) to the TRACE operation. Also added a newline in that TRACE statement,
  which was missing.

   \version 1.46 (12 Apr 2002)
  fixed an obscure bug in SetPixelMatrix: when working with overlapped tiles, we do not
  really want to copy the whole tile back into the image, but only the central section.
  The exception is on the image borders, where we actually will be copying back data
  from the original image.

   \version 1.47 (29 Apr 2002)
  made DisplayProgress operations const functions - otherwise this would not allow
  their use in otherwise const classes.

   \version 1.48 (4 May 2002)
  fixed a bug that applies to 16-bit modes: in Photoshop it seems that the 16-bit
  modes actually store pixels in the range 0->32768 rather than 0->32767 as I thought
  (or 0->65535 as they should!). This was noted by L. Chang in an email to the SDK
  mailing list and confirmed by Tom Ruark on 05/02/02. Note also that this class
  converts from integers to floating point and vice versa only by multiplication or
  division - no boolean operations are performed; this allows the derived plugins
  to manipulate beyond the boundary if necessary. Also modified the conversion from
  double to int (for 8 & 16-bit) to do a rounding operation rather than straight
  conversion (which might truncate data).

   \version 1.50 (1 Nov 2002)
  added functions for scripting support - since pFilterRecord was made private in
   version 1.32, derived classes need a way to access the Descriptor Suite. The
  parameters are read from scripting during FilterStart, and written back during
  FilterFinish; also made GetShowDialog a const member.

   \version 1.51 (2 Nov 2002)
  added more functions for scripting support - in order to simplify reading/writing
  of descriptor key/value pairs, we added functions to do so, and convert between
  STL or MFC types (which we usually use) and the Adobe SDK types required by the
  scripting system; also changed low-level scripting routines to private, and made
  low-level read/write script parameters non-virtual.

   \version 1.52 (8 Nov 2002)
  added Main function which contains the standard PluginMain routine for filter
  plugins - this reduces the amount of copying between different plugins and ensures
  better uniformity.

   \version 1.53 (8 Nov 2002)
  bug fix: moved memory allocation for data block to a new private function, which is
  now called from both FilterPrepare and FilterParameters, as necessary, since the
  latter isn't called when the filter is invoked through the scripting system. Thus,
  if the filter is invoked through scripting without first being invoked (at least
  once) directly by the user, then memory would never be allocated, causing an error.

   \version 1.54 (12 Nov 2002)
  added tracing for read/write script parameters and data allocation; removed data
  allocation from FilterParameters, since it is totally redundant there.

   \version 1.55 (19 Feb 2003)
  added definition for DLLExport - this was previously being taken from PIDefines, which
  is part of the SDK sample code and not of the API itself. This removes all need for
  including files from the sample code.

   \version 1.60 (6 Nov 2006)
   - defined class and associated data within "libwin" namespace.
   - removed pragma once directive, as this is unnecessary
   - changed unique define to conform with that used in other libraries
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.61 (10 Nov 2006)
   - made class a derivative of CRoutedIO.

   \version 1.62 (7 Nov 2007)
   - moved Adobe SDK includes here from stdafx.h.
   - made this module compile only when ADOBESDK is defined.
*/

namespace libwin {

class CPSPlugIn : public CRoutedIO
{
private:
   int            m_nDataSize;      // size of data block
   int            m_nDataVersion;   // version of data block (indicate changed contents)
   bool           m_bShowDialog;    // true to show user-interface dialog
   int16          m_nPlane;         // current tile - plane
   int16          m_nHorTile;       // current tile - horizontal index
   int16          m_nVerTile;       // current tile - vertical index
   int16          m_nTileOverlap;   // pixel overlap between tiles
   libbase::timer m_tDuration;      // operation timer

   FilterRecord*  m_pFilterRecord;
   long*          m_pData;

private:
   // data block memory allocation
   void DataAllocate();
   void DataCheckVersion();

   // tile manipulation
   void TileCopy();
   void TileSetEmpty();
   void TileWrite();
   void TileUpdate();

   // descriptor suite - low-level access
   PIDescriptorParameters* GetDescParams() { return m_pFilterRecord->descriptorParameters; };
   WriteDescriptorProcs* GetWriteDescProcs() { return GetDescParams()->writeDescriptorProcs; };
   ReadDescriptorProcs* GetReadDescProcs() { return GetDescParams()->readDescriptorProcs; };
   void WriteScriptParameters();
   void ReadScriptParameters();

protected:
   // current tile information
   int GetCurrentTileVer() const { return m_nVerTile; };
   int GetCurrentTileHor() const { return m_nHorTile; };
   int GetCurrentPlane() const { return m_nPlane; };
   int GetCurrentCoordLeft() const { return m_pFilterRecord->outRect.left; };
   int GetCurrentCoordRight() const { return m_pFilterRecord->outRect.right; };
   int GetCurrentCoordTop() const { return m_pFilterRecord->outRect.top; };
   int GetCurrentCoordBottom() const { return m_pFilterRecord->outRect.bottom; };
   int GetCurrentTileWidth() const { return m_pFilterRecord->outRect.right - m_pFilterRecord->outRect.left; };
   int GetCurrentTileHeight() const { return m_pFilterRecord->outRect.bottom - m_pFilterRecord->outRect.top; };

   // user setup & info functions
   void SetTileOverlap(const int nOverlap) { m_nTileOverlap = nOverlap; };
   void SetTileWidth(const int nTileWidth) { m_pFilterRecord->outTileWidth = nTileWidth; };
   void SetTileHeight(const int nTileHeight) { m_pFilterRecord->outTileHeight = nTileHeight; };

   void SetShowDialog(const bool bShowDialog) { m_bShowDialog = bShowDialog; };
   bool GetShowDialog() const { return m_bShowDialog; };

   // user hooks for multi-pass operators
   bool IterationDone() const;
   void IterationStart();

   // user progress display functions
   void DisplayTotalProgress(const int nComplete, const int nTotal=100, const int nIteration=0, const int nTotalIterations=1) const;
   void DisplayTileProgress(const int nComplete, const int nTotal=100, const int nIteration=0, const int nTotalIterations=1) const;

   // pixel conversion
   void GetPixelMatrix(libbase::matrix<double>& m);
   void SetPixelMatrix(const libbase::matrix<double>& m);
   double GetPixelValue(const int x, const int y);
   void SetPixelValue(const int x, const int y, const double c);

   // descriptor suite - utilities
   bool IsDescriptorAvailable() const { return m_pFilterRecord->descriptorParameters != NULL; };
   // descriptor suite - write
   PIWriteDescriptor OpenWriter() { return GetWriteDescProcs()->openWriteDescriptorProc(); };
   void CloseWriter(PIWriteDescriptor token);
   void PutString(PIWriteDescriptor token, DescriptorKeyID key, const char *data);
   void PutInteger(PIWriteDescriptor token, DescriptorKeyID key, int data);
   void PutFloat(PIWriteDescriptor token, DescriptorKeyID key, double data);
   void PutBoolean(PIWriteDescriptor token, DescriptorKeyID key, bool data);
   // descriptor suite - read
   PIReadDescriptor OpenReader(DescriptorKeyIDArray array=NULL) { return GetReadDescProcs()->openReadDescriptorProc(GetDescParams()->descriptor, array); };
   void CloseReader(PIReadDescriptor token);
   bool GetString(PIReadDescriptor token, char *data);
   bool GetInteger(PIReadDescriptor token, int *data);
   bool GetFloat(PIReadDescriptor token, double *data);
   bool GetBoolean(PIReadDescriptor token, bool *data);

   // virtual overrides - scripting
   virtual void WriteScriptParameters(PIWriteDescriptor token) {};
   virtual void ReadScriptParameter(PIReadDescriptor token, DescriptorKeyID key, DescriptorTypeID type, int32 flags) {};

   // virtual overrides - other
   virtual void ShowDialog(void) = 0;
   virtual void InitPointer(char* sData) = 0;
   virtual void InitParameters() = 0;

public:
   // image information
   bool IsWide() const { return (m_pFilterRecord->depth == 16); };
   int GetPlanes() const { return (m_pFilterRecord->filterCase > 2) ? m_pFilterRecord->outLayerPlanes : m_pFilterRecord->planes; };
   int GetImageWidth() const { return m_pFilterRecord->imageSize.h; };
   int GetImageHeight() const { return m_pFilterRecord->imageSize.v; };

   // tiling information
   int GetTilesHor() const { return int(ceil(GetImageWidth() / double(GetSuggestedTileWidth() - m_nTileOverlap))); };
   int GetTilesVer() const { return int(ceil(GetImageHeight() / double(GetSuggestedTileHeight() - m_nTileOverlap))); };
   int GetSuggestedTileWidth() const { return m_pFilterRecord->outTileWidth; };
   int GetSuggestedTileHeight() const { return m_pFilterRecord->outTileHeight; };

public:
   // creation / destruction
        CPSPlugIn(const int nDataSize, const int nDataVersion);
        virtual ~CPSPlugIn();

   // plug-in main function
   void Main(const short nSelector, FilterRecord* pFilterRecord, long* pData, short* pResult);

   // plug-in entry & exit
   void Entry(FilterRecord* pFilterRecord, long* pData);
   void Exit();

   // plug-in interface
   virtual void FilterAbout(void);
   virtual void FilterParameters(void);
   virtual void FilterPrepare(void);
   virtual void FilterStart(void);
   virtual void FilterContinue(void);
   virtual void FilterFinish(void);
};

} // end namespace

#endif

#endif
