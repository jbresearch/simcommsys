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

#ifndef __resultsfile_factory_h
#define __resultsfile_factory_h

#include "assertalways.h"
#include "resultsfile.h"
#include "resultsfile_json.h"
#include "resultsfile_text.h"

#include <memory>
#include <string>

namespace libcomm
{

class montecarlo;

/*! \brief factory to return the desired resultsfile implementation
 * This factory allows the user to choose between concrete subclasses of
 * resultsfile, allowing for a selection of formats.
 *
 * text is the old interface which writes the output file in a specialized text
 * format.
 */
class resultsfile_factory
{
public:
    static std::unique_ptr<resultsfile> get_resultsfile(const std::string type,
                                                        montecarlo& simulator)
    {
        std::unique_ptr<resultsfile> resultsfile_ptr;

        if (type == "text") {
            resultsfile_ptr = std::make_unique<resultsfile_text>(simulator);
        } else if (type == "json") {
            resultsfile_ptr = std::make_unique<resultsfile_json>(simulator);
        } else {
            std::string error_msg(type + " is not a valid SPA type");
            failwith(error_msg.c_str());
        }

        return resultsfile_ptr;
    }
};

} // namespace libcomm

#endif // __resultsfile_factory_h