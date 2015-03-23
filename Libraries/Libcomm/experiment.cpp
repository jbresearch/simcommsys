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

#include "experiment.h"

namespace libcomm {

using libbase::vector;

void experiment::prettyprint_results(std::ostream& sout, const libbase::vector<
      double>& result, const libbase::vector<double>& errormargin) const
   {
   const int N = result.size();
   for (int i = 0; i < N; i++)
      {
      sout << result_description(i) << '\t';
      sout << result(i) << '\t';
      sout << "[±" << errormargin(i) << " = ";
      sout << fabs(100 * errormargin(i) / result(i)) << "%]" << std::endl;
      }
   }

} // end namespace
