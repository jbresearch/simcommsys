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

/*!
 * \file
 * \brief   CUDA matrix in device memory.
 * \author  Johann Briffa
 */

#include "cuda-all.h"
#include "gf.h"

namespace cuda {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define TYPE_SEQ \
   (bool)(int)(float)(double) \
   GF_TYPE_SEQ

#define INSTANTIATE(r, x, type) \
      template class matrix<type>; \
      template class matrix<vector<type> >; \
      template class matrix<vector_reference<type> >; \
      template class matrix<vector_auto<type> >;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, TYPE_SEQ)

} // end namespace
