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

#ifndef __resultsfile_h
#define __resultsfile_h

#include "config.h"

#include "timer.h"
#include "sha.h"
#include <iostream>

namespace libcomm {

/*!
 * \brief   Results File Handler.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class resultsfile {
private:
   /*! \name User-specified parameters */
   std::string fname; //!< Filename for associated results file
   // @}
   /*! \name Internal variables */
   bool headerwritten; //!< Flag to indicate that the results header has been written
   std::streampos fileptr; //!< Position in file where we should write the next result
   sha filedigest; //!< Digest of file as at last update
   libbase::timer t; //!< Timer to keep track of running estimate
   // @}
private:
   /*! \name Results file helper functions */
   void finishwithfile(std::fstream& file);
   void truncate(std::streampos length);
   void checkformodifications(std::fstream& file);
   // @}
protected:
   /*! \name System-specific functions */
   virtual void writeheader(std::ostream& sout) const = 0;
   virtual void
         writeresults(std::ostream& sout, libbase::vector<double>& result,
               libbase::vector<double>& tolerance) const = 0;
   virtual void writestate(std::ostream& sout) const = 0;
   virtual void lookforstate(std::istream& sin) = 0;
   // @}
public:
   /*! \name Constructor/destructor */
   resultsfile();
   virtual ~resultsfile();
   // @}

   /*! \name File handling interface */
   void init(const std::string& fname);
   bool isinitialized() const
      {
      return !fname.empty();
      }
   // @}

   /*! \name Results handling interface */
   void setupfile();
   void writeinterimresults(libbase::vector<double>& result, libbase::vector<
         double>& tolerance);
   void writefinalresults(libbase::vector<double>& result, libbase::vector<
         double>& tolerance, bool savestate = false);
   // @}
};

} // end namespace

#endif
