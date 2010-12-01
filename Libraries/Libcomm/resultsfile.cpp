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
 * 
 * \section svn Version Control
 * - $Id$
 */

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "resultsfile.h"

#include <fstream>

#ifdef WIN32
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
#ifdef WIN32
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
   trace << "DEBUG (montecarlo): checking file for modifications." << std::endl;
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

// Constructor/destructor

resultsfile::resultsfile() :
   t("resultsfile")
   {
   headerwritten = false;
   t.stop();
   }

resultsfile::~resultsfile()
   {
   assert(!t.isrunning());
   }

// File handling interface

void resultsfile::init(const std::string& fname)
   {
   assert(!t.isrunning());
   headerwritten = false;
   resultsfile::fname = fname;
   }

// Results handling interface

void resultsfile::setupfile()
   {
   assert(!fname.empty());
   // start timer for interim results writing
   t.start();
   // if this is not the first time, that's all
   if (headerwritten)
      return;
   // open file for input and output
   std::fstream file(fname.c_str());
   if (!file)
      {
      trace << "DEBUG (montecarlo): results file not found - creating." << std::endl;
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
   // write header at end
   file.seekp(0, std::ios_base::end);
   writeheader(file);
   // update write-position
   fileptr = file.tellp();
   // update digest
   file.seekg(0);
   filedigest.process(file);
   // update flags
   headerwritten = true;
   }

void resultsfile::writeinterimresults(libbase::vector<double>& result,
      libbase::vector<double>& tolerance)
   {
   assert(!fname.empty());
   assert(t.isrunning());
   // restrict updates to occur every 30 seconds or less
   if (t.elapsed() < 30)
      return;
   // open file for input and output
   std::fstream file(fname.c_str());
   assertalways(file.good());
   checkformodifications(file);
   writeresults(file, result, tolerance);
   writestate(file);
   finishwithfile(file);
   // restart timer
   t.start();
   }

void resultsfile::writefinalresults(libbase::vector<double>& result,
      libbase::vector<double>& tolerance, bool savestate)
   {
   assert(!fname.empty());
   assert(t.isrunning());
   // open file for input and output
   std::fstream file(fname.c_str());
   assertalways(file.good());
   checkformodifications(file);
   writeresults(file, result, tolerance);
   if (savestate)
      writestate(file);
   // update write-position
   fileptr = file.tellp();
   finishwithfile(file);
   // stop timer
   t.stop();
   }

} // end namespace
