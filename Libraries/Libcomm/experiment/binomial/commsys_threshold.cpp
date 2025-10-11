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

#include "commsys_threshold.h"

#include <sstream>

namespace libcomm
{

// Experiment parameter handling

template <class S>
void
commsys_threshold<S>::set_parameters(const libbase::vector<double>& params)
{
    assertalways(params.size() == 1);
    mono_parametric& m = dynamic_cast<mono_parametric&>(*this->sys->getmodem());
    m.set_parameter(params(0));
}

template <class S>
libbase::vector<double>
commsys_threshold<S>::get_parameters() const
{
    const mono_parametric& m =
        dynamic_cast<const mono_parametric&>(*this->sys->getmodem());
    libbase::vector<double> params;
    params.init(1);
    params(0) = m.get_parameter();
    return params;
}

// Description & Serialization

template <class S>
std::string
commsys_threshold<S>::description() const
{
    std::ostringstream sout;
    sout << "Modem-threshold-varying ";
    sout << Base::description();
    return sout.str();
}

template <class S>
std::ostream&
commsys_threshold<S>::serialize(std::ostream& sout) const
{
    sout << Base::get_parameters() << std::endl;
    Base::serialize(sout);
    return sout;
}

template <class S>
std::istream&
commsys_threshold<S>::serialize(std::istream& sin)
{
    libbase::vector<double> params;
    sin >> libbase::eatcomments >> params >> libbase::verify;
    Base::serialize(sin);
    Base::set_parameters(params);
    return sin;
}

} // namespace libcomm

#include "gf.h"
#include "result_collector/commsys/errors_hamming.h"
#include "result_collector/commsys/errors_levenshtein.h"
#include "result_collector/commsys/hist_symerr.h"
#include "result_collector/commsys/prof_burst.h"
#include "result_collector/commsys/prof_pos.h"
#include "result_collector/commsys/prof_sym.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ

/* Serialization string: commsys_threshold<type,collector>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      collector = errors_hamming | errors_levenshtein | ...
 */
#define INSTANTIATE(r, x, type) \
      template class commsys_threshold<type>; \
      template <> \
      const serializer commsys_threshold<type>::shelper( \
            "experiment", \
            "commsys_threshold<" BOOST_PP_STRINGIZE(type) ">", \
            commsys_threshold<type>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // namespace libcomm
