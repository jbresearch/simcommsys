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

#include "assertalways.h"
#include "resultsfile.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

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
private:
    /*! \name System-specific functions */
    /*! \brief Writes results for a specific parameter to output file in JSON
     * format. */
    void writeresults(std::fstream& sout,
                      libbase::vector<double>& result,
                      libbase::vector<double>& errormargin) const;
    /*! \brief Writes state to output file in JSON format. */
    void writestate(std::fstream& sout) const;
    /*! \brief Writes metadata to the output file, such as date and Simcommsys
     * version. */
    void writemetadata(std::fstream& sout) const;
    /*! \brief Determines if metadata needs to be written to output file;
     * metadata should only be written once to an output file.
     *
     * \note At the moment this function simply checks if the file is empty or
     * not. If empty it writes metadata.
     */
    void writemetadataifneeded(std::fstream& sout) const;
    // @}

    /*! \name Helper methods for reading/writing JSON data to the files. */
    /*! \brief Read JSON data in \p sout if any and return it (empty JSON object
     * is returned if file is empty)
     */
    nlohmann::json readjson(std::fstream& sout) const;
    /*! \brief Write JSON \p data to file specified by \p sout
     * \note Contents of \p sout are truncated when writing.
     */
    void writejson(std::fstream& sout, const nlohmann::json& data) const;
    // @}

protected:
    /*! \name System-specific functions */
    void lookforstate(std::fstream& sin) override;
    /*! \brief Write current results and perhaps the state
     */
    void writeresultsandstate(std::fstream& file,
                              libbase::vector<double>& result,
                              libbase::vector<double>& errormargin,
                              bool savestate,
                              bool interim) override
    {
        if (this->wasmodified(file))
            failwith(
                "Output file cannot be modified during simulation when using "
                "JSON format.");

        writemetadataifneeded(file);
        writeresults(file, result, errormargin);
        if (savestate)
            writestate(file);
    };
    // @}
public:
    using resultsfile::resultsfile;
};
} // namespace libcomm

#endif // __resultsfile_json_h