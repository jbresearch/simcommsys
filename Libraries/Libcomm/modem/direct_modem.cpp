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

#include "direct_modem.h"
#include <cstdlib>
#include <sstream>

namespace libcomm
{

// *** Templated GF(q) modem ***

// Description

template <class G>
std::string
direct_modem<G>::description() const
{
    std::ostringstream sout;
    sout << "GF(" << num_symbols() << ") Modulation";
    return sout.str();
}

// *** Specific to direct_modem_implementation<bool> ***

// Description

std::string
direct_modem<bool>::description() const
{
    return "Binary Modulation";
}

} // namespace libcomm

#include "erasable.h"
#include "gf.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>

using libbase::erasable;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

#define ADD_ERASABLE(r, x, type) \
   (type)(erasable<type>)

#define ALL_GF_TYPE_SEQ \
   BOOST_PP_SEQ_FOR_EACH(ADD_ERASABLE, x, GF_TYPE_SEQ)

#define INSTANTIATE(r, x, type) \
      template class direct_modem<type>;
// clang-format on

// NOTE: we need to avoid instantiating direct_modem<bool>
// because this is an explicit specialization

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, (erasable<bool>)ALL_GF_TYPE_SEQ)

} // namespace libcomm
