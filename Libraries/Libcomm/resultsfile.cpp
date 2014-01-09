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

#include "resultsfile.h"

#include <fstream>

#ifdef _WIN32
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#else
#  include <unistd.h>
#  include <sys/types.h>
#endif

namespace libcomm {

using std::cerr;
using libbase::trace;
using libbase::vector;

// Results file helper functions

/*! \brief If this is the first time, write the header
 * \note This method also updates the write position so that the header is not
 * overwritten on the next write.
 */
void resultsfile::writeheaderifneeded(std::fstream& file)
   {
   if (!headerwritten)
      {
      writeheader(file);
      // update flag
      headerwritten = true;
      // update file-write position
      fileptr = file.tellp();
      }
   }

/*! \brief Close and truncate the file, and update digest
 * Truncation is needed to remove any detritus from previously-saved states.
 */
void resultsfile::finishwithfile(std::fstream& file)
   {
   std::streampos length = file.tellp();
   // close and truncate file
   file.close();
   truncate(length);
   // re-open and update digest
   file.open(fname.c_str(), std::ios::in);
   filedigest.process(file);
   }

void resultsfile::truncate(std::streampos length)
   {
   assert(!fname.empty());
#ifdef _WIN32
   int fd;
   _sopen_s(&fd, fname.c_str(), _O_RDWR, _SH_DENYNO, _S_IREAD | _S_IWRITE);
   _chsize_s(fd, length);
   _close(fd);
#else
   assertalways(::truncate(fname.c_str(), length)==0);
#endif
   }

void resultsfile::checkformodifications(std::fstream& file)
   {
   assert(file.good());
   trace << "DEBUG (resultsfile): checking file for modifications." << std::endl;
   // check for user modifications
   sha curdigest;
   file.seekg(0);
   curdigest.process(file);
   // reset file
   file.clear();
   if (curdigest == filedigest)
      file.seekp(fileptr);
   else
      {
      cerr << "NOTICE: file modifications found - appending." << std::endl;
      // set current write position to end-of-file
      file.seekp(0, std::ios_base::end);
      fileptr = file.tellp();
      }
   }

// File handling interface

void resultsfile::init(const std::string& fname)
   {
   assert(!t.isrunning());
   filesetup = false;
   headerwritten = false;
   resultsfile::fname = fname;
   }

// Results handling interface

/*! \brief Set up the results file and look for a state
 * If the file does not exist, a new one is created. Otherwise, the write
 * point is set to the end of file and a digest of the current file contents
 * is kept. A search for a saved state is also initiated by this method.
 *
 * \note The current simulation must be already set up at this point, so that
 * a valid comparison can be made.
 */
void resultsfile::setupfile()
   {
   assert(!fname.empty());
   assert(!filesetup);
   // open file for input and output
   std::fstream file(fname.c_str());
   if (!file)
      {
      trace << "DEBUG (resultsfile): results file not found - creating." << std::endl;
      // create the file first if necessary
      file.open(fname.c_str(), std::ios::out);
      assertalways(file.good());
      // then reopen for input/output
      file.close();
      file.open(fname.c_str());
      }
   assertalways(file.good());
   // look for saved-state
   lookforstate(file);
   // set write position at end
   file.seekp(0, std::ios_base::end);
   fileptr = file.tellp();
   // update digest
   file.seekg(0);
   filedigest.process(file);
   // start timer for interim results writing
   t.start();
   // update flags
   filesetup = true;
   }

/*! \brief Write current results and state
 * This method can be called as many times as required; usually this is
 * called after every update. File writes are limited to occur no more often
 * than 30 seconds (this quantity is hard-wired).
 *
 * \note This method does not change the write position so that this result is
 * overwritten on the next write.
 */
void resultsfile::writeinterimresults(libbase::vector<double>& result,
      libbase::vector<double>& errormargin)
   {
   assert(filesetup);
   assert(t.isrunning());
   // restrict updates to occur every 30 seconds or less
   if (t.elapsed() < 30)
      return;
   // open file for input and output
   std::fstream file(fname.c_str());
   assertalways(file.good());
   checkformodifications(file);
   writeheaderifneeded(file);
   writeresults(file, result, errormargin);
   writestate(file);
   finishwithfile(file);
   // restart timer
   t.start();
   }

/*! \brief Write final results and state
 * This method is called when the final result is reached. A file write is
 * guaranteed to occur. The write-limiting timer is also stopped to avoid
 * lapsing on object destruction. If requested, the final state is also
 * written.
 *
 * \note This method also updates the write position so that this result is not
 * overwritten.
 */
void resultsfile::writefinalresults(libbase::vector<double>& result,
      libbase::vector<double>& errormargin, bool savestate)
   {
   assert(filesetup);
   assert(t.isrunning());
   // open file for input and output
   std::fstream file(fname.c_str());
   assertalways(file.good());
   checkformodifications(file);
   writeheaderifneeded(file);
   writeresults(file, result, errormargin);
   if (savestate)
      writestate(file);
   // update write-position
   fileptr = file.tellp();
   finishwithfile(file);
   // stop timer and clear setup flag (in preparation for next simulation run)
   t.stop();
   filesetup = false;
   }

} // end namespace
