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

#ifndef __resultsfile_h
#define __resultsfile_h

#include "config.h"

#include "walltimer.h"
#include "sha.h"
#include <iostream>

namespace libcomm {

/*!
 * \brief   Results File Handler.
 * \author  Johann Briffa
 *
 * This class encapsulates the process of writing results to a file. It opens
 * and closes the file for every write, ensuring that written results are
 * flushed, and also allowing the user to manipulate the file between writes.
 *
 * The handler keeps track of the file digest between writes, checking for
 * any external changes. In such cases, the file is considered 'modified'
 * and the next write happens at the end of the file.
 *
 * The handler also allows 'interim' result writing. In this case, the result
 * is written together with the simulation state. This allows the user to
 * continue an aborted simulation (due to simulator or machine crash, for
 * example). When a file is initialized, a search for the last-save simulator
 * state is performed. If this matches the current system at the current
 * parameter, this state needs to be loaded.
 *
 * \note The handler does not specify the format for writing any of the
 * header, result lines, or state. Instead, these functions are performed
 * by virtual methods.
 *
 * Classes implementing this interface need to:
 * 1) provide implementations for virtual methods (writing header, results
 *    and state, and seeking + reading state)
 * 2) use the interface as follows:
 *    a) init() to provide filename, after which the following may be used:
 *    b) setupfile() to look for a state; the current simulation must be
 *       already set up at this point, so that a valid comparison can be made.
 *    c) writeinterimresults() as many times as required; usually this is
 *       called after every update. The handler limits file writes to occur
 *       no more often than 30 seconds.
 *    d) writefinalresults() one last time; this is guaranteed to happen.
 */

class resultsfile {
private:
   /*! \name User-specified parameters */
   std::string fname; //!< Filename for associated results file
   // @}
   /*! \name Internal variables */
   bool filesetup; //!< Flag to indicate that the results file was set up
   bool headerwritten; //!< Flag to indicate that the results header has been written
   std::streampos fileptr; //!< Position in file where we should write the next result
   sha filedigest; //!< Digest of file as at last update
   libbase::walltimer t; //!< Timer to keep track of running estimate
   // @}
private:
   /*! \name Results file helper functions */
   void writeheaderifneeded(std::fstream& file);
   void finishwithfile(std::fstream& file);
   void truncate(std::streampos length);
   void checkformodifications(std::fstream& file);
   // @}
protected:
   /*! \name System-specific functions */
   virtual void writeheader(std::ostream& sout) const = 0;
   virtual void
   writeresults(std::ostream& sout, libbase::vector<double>& result,
         libbase::vector<double>& errormargin) const = 0;
   virtual void writestate(std::ostream& sout) const = 0;
   virtual void lookforstate(std::istream& sin) = 0;
   // @}
public:
   /*! \name Constructor/destructor */
   // Constructor/destructor
   resultsfile() :
      filesetup(false), headerwritten(false), t("resultsfile", false)
      {
      }
   virtual ~resultsfile()
      {
      assert(!t.isrunning());
      }
   // @}

   /*! \name File handling interface */
   /*! \brief Provide filename
    * After this, the results handling interface methods can be used.
    */
   void init(const std::string& fname);
   /*! \brief Check whether the handler has been initialized
    * Indicates whether the results handling interface methods can be used.
    */
   bool isinitialized() const
      {
      return !fname.empty();
      }
   // @}

   /*! \name Results handling interface */
   void setupfile();
   void writeinterimresults(libbase::vector<double>& result, libbase::vector<
         double>& errormargin);
   void writefinalresults(libbase::vector<double>& result, libbase::vector<
         double>& errormargin, bool savestate = false);
   // @}
};

} // end namespace

#endif
