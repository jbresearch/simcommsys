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

#ifndef __resultsfile_json_h
#define __resultsfile_json_h

#include "resultsfile.h"
#include <iostream>

namespace libcomm
{

/*!
 * \brief   Results JSON File Handler.
 * \author  Mark Mizzi
 *
 * This class encapsulates the process of writing results to a file in the
 * JSON format. The user is not allowed to manipulate the file between
 * writes; detecting an external modification will trigger a fatal error.
 * External modifications are tracked by \ref resultsfile using the
 * resultsfile::filedigest and associated mechanisms.
 */
class resultsfile_json : public resultsfile
{
protected:
    /*! \name System-specific functions */
    void lookforstate(std::istream& sin) override;
    void writeresultsandstate(std::fstream& file,
                              libbase::vector<double>& result,
                              libbase::vector<double>& errormargin,
                              bool savestate,
                              bool interim) override;
    // @}
public:
    using resultsfile::resultsfile;
};
} // namespace libcomm

#endif // __resultsfile_json_h