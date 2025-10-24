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

template <class S, class T>
void
qkd_commsys_simulator<S, T>::sample(libbase::vector<double>& sample_result,
                             libbase::vector<uint64_t>& sample_count)
{
      // Reset timers
    this->reset_timers();
    // Initialise sample result and count vectors
    sample_result.init(result_count());
    sample_count.init(result_count());
    sample_result = 0;
    sample_count = 0;

    // Gets the number of coherent states generated for a single frame from the
    // qkd_commsys object.
    const libbase::size_type<libbase::vector> framesize(
        sys->input_block_size());

    // Generates a sequence of coherent states which is the input to the
    // fullcycle method in qkd_commsys.
    libbase::vector<S> source = src->generate_sequence(framesize);

    // Both final keys are of libbase::vector<bool> type.
    auto [key_KA, key_KB] = sys->fullcycle(source);

    libbase::indirect_vector<double> sample_result_segment =
        sample_result.segment(0, rc->result_count());
    libbase::indirect_vector<uint64_t> sample_count_segment =
        sample_count.segment(0, rc->result_count());

    // CV collector
    rc->compute_result_and_accumulate(sample_result_segment, sample_count_segment, key_KA, key_KB);
}

template <class S, class T>
std::string
qkd_commsys_simulator<S, T>::description() const
{
    std::ostringstream sout;
    sout << "QKD Simulator for ";
    sout << sys->description();
    sout << ", ";
    sout << src->description();
    sout << ", collecting ";
    sout << rc->description();
    return sout.str();
}

// object serialization - saving

template <class S, class T>
std::ostream&
qkd_commsys_simulator<S, T>::serialize(std::ostream& sout) const
{
    // format version
    sout << "# Version" << std::endl;
    sout << 1 << std::endl;
    sout << "# Results collector" << std::endl;
    sout << rc;
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
template <class S, class T>
std::istream&
qkd_commsys_simulator<S, T>::serialize(std::istream& sin)
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
    sin >> libbase::eatcomments >> rc >> libbase::verify;

    assertalways(sin.good());
    return sin;
}

} // namespace libcomm

#include "result_collector/qkd_commsys/cv_qkd_errors_hamming.h"
#include "result_collector/qkd_commsys/dv_qkd_errors_hamming.h"
// #include "result_collector/commsys/errors_hamming.h"

namespace libcomm
{

// ----- explicit instantiations & serializer registration (Boost PP) -----
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>

// clang-format off
#define STATE_SEQ (gaussian_state)(qubit)
#define SCALAR_SEQ (double)(bool)
// clang-format on

/* Serialization string: qkd_commsys_simulator<S,T>
 * where:
 *      S = gaussian_state | qubit
 *      T = double | bool
 */

#define INSTANTIATE(r, args)                                                           \
    template class qkd_commsys_simulator<BOOST_PP_SEQ_ENUM(args)>;                     \
    template <>                                                                        \
    const libbase::serializer qkd_commsys_simulator<BOOST_PP_SEQ_ENUM(args)>::shelper( \
        "experiment",                                                                  \
        "qkd_commsys_simulator<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, args)) "," BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1, args)) ">",          \
                                                qkd_commsys_simulator<                 \
                                                    BOOST_PP_SEQ_ENUM(                 \
                                                        args)>::create);

// BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
//                               (STATE_SEQ)(SCALAR_SEQ)(COLLECTOR_TYPE_SEQ))

// Instantiate the serializers for the only valid combinations

INSTANTIATE(0, (gaussian_state)(double)) // CV
INSTANTIATE(0, (qubit)(bool))            // DV

} // namespace libcomm