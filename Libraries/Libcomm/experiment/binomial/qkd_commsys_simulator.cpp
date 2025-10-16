/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi, Aaron Abela
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

#include "qkd_commsys_simulator.h"

#include <string>

namespace libcomm
{

template <class S, class T, class R>
void
qkd_commsys_simulator<S, T, R>::sample(array1d_t& result)
{
    // Reset timers
    this->reset_timers();
    // Initialise result vector
    result.init(count());
    result = 0;

    // Gets the number of coherent states generated for a single frame from the
    // qkd_commsys object.
    const libbase::size_type<libbase::vector> framesize(
        sys->input_block_size());

    // Generates a sequence of coherent states which is the input to the
    // fullcycle method in qkd_commsys.
    libbase::vector<S> source = src->generate_sequence(framesize);

    // qkd_commsys sends the src generator to get VA for CV-QKD.
    sys->set_src(src);

    // Both final keys are of libbase::vector<bool> type.
    auto [key_KA, key_KB] = sys->fullcycle(source);

    libbase::indirect_vector<double> result_segment =
        result.segment(0, R::count());

    // CV collector
    R::updateresults(result_segment, source, key_KA, key_KB);
}

template <class S, class T, class R>
std::string
qkd_commsys_simulator<S, T, R>::description() const
{
    std::ostringstream sout;
    sout << "QKD Simulator for ";
    sout << sys->description();
    sout << ", ";
    sout << src->description();
    return sout.str();
}

// object serialization - saving

template <class S, class T, class R>
std::ostream&
qkd_commsys_simulator<S, T, R>::serialize(std::ostream& sout) const
{
    // format version
    sout << "# Version" << std::endl;
    sout << 1 << std::endl;
    sout << "# Source generator" << std::endl;
    sout << src;
    sout << "# Communication system" << std::endl;
    sout << sys;
    return sout;
}

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering; added split channel model
 */
template <class S, class T, class R>
std::istream&
qkd_commsys_simulator<S, T, R>::serialize(std::istream& sin)
{
    assertalways(sin.good());

    // get format version
    int version;
    sin >> libbase::eatcomments >> version;

    // handle old-format files
    if (sin.fail()) {
        version = 0;
        sin.clear();
    }

    sin >> libbase::eatcomments >> src >> libbase::verify;
    sin >> libbase::eatcomments >> sys >> libbase::verify;

    assertalways(sin.good());
    return sin;
}

} // namespace libcomm

#include "result_collector/qkd_commsys/cv_qkd_errors_hamming.h"
// #include "result_collector/commsys/errors_hamming.h"

namespace libcomm
{

// ----- explicit instantiations & serializer registration (Boost PP) -----
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>

// clang-format off
#define STATE_SEQ (gaussian_state)
#define SCALAR_SEQ (double)
#define COLLECTOR_TYPE_SEQ (cv_qkd_errors_hamming)

/* Serialization string qkd_commsys_simulator<S, T, R>:
 * qkd_commsys_simulator<gaussian_state, double, cv_qkd_errors_hamming> where:
 *  S (type of quantum state) = gaussian_state ..
 *  T (type) = double, float, bool.
 *  R (results collector) = cv_qkd_errors_hamming
 */

#define INSTANTIATE(r, args) \
    template class qkd_commsys_simulator<BOOST_PP_SEQ_ENUM(args)>; \
    template <> \
    const libbase::serializer qkd_commsys_simulator<BOOST_PP_SEQ_ENUM(args)>::shelper( \
        "experiment", \
        "qkd_commsys_simulator<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, args)) "," \
        BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1, args)) "," \
        BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2, args)) ">", \
        qkd_commsys_simulator<BOOST_PP_SEQ_ENUM(args)>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
                              (STATE_SEQ)(SCALAR_SEQ)(COLLECTOR_TYPE_SEQ))

} // namespace libcomm