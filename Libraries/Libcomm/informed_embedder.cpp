/*!
 * \file
 *
 * Copyright (c) 2025 Johann A. Briffa
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

#include "informed_embedder.h"
#include <cstdlib>
#include <sstream>

namespace libcomm
{

// *** Common Data Embedder/Extractor Interface ***

// Explicit Realizations

template class basic_informed_embedder<int>;
template class basic_informed_embedder<float>;
template class basic_informed_embedder<double>;

} // namespace libcomm
