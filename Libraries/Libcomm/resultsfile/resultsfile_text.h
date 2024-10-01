/*!
 * \file
 *
 * Copyright (c) 2024 Mark Mizzi
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

#ifndef __resultsfile_text_h
#define __resultsfile_text_h

#include "resultsfile.h"
#include <iostream>

namespace libcomm
{

class resultsfile_text : public resultsfile
{
private:
    std::streampos
        fileptr; //!< Position in file where we should write the next result

protected:
    /*! \name System-specific functions */
    void writeheader(std::ostream& sout) const;
    void writeresults(std::ostream& sout,
                      libbase::vector<double>& result,
                      libbase::vector<double>& errormargin) const;
    void writestate(std::ostream& sout) const;
    void lookforstate(std::istream& sin) override;
    void writeresults(std::fstream& file,
                      libbase::vector<double>& result,
                      libbase::vector<double>& errormargin,
                      bool savestate,
                      bool interim) override;
    // @}

    /*! \name Results file text helper functions */
    void writeheaderifneeded(std::fstream& file);
    void checkformodifications(std::fstream& file);
    // @}

public:
    using resultsfile::resultsfile;

    void setupfile() override;
};
} // namespace libcomm

#endif // __resultsfile_text_h