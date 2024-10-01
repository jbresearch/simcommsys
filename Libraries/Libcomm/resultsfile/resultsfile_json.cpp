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

#include "resultsfile_json.h"
#include "montecarlo.h"
#include "vector.h"
#include "version.h"

#include <nlohmann/json.hpp>

namespace libcomm
{

void
resultsfile_json::lookforstate(std::istream& sin)
{
}

/*! \brief Write current results and perhaps the state
 */
void
resultsfile_json::writeresultsandstate(std::fstream& file,
                                       libbase::vector<double>& result,
                                       libbase::vector<double>& errormargin,
                                       bool savestate,
                                       bool interim)
{
}

} // namespace libcomm