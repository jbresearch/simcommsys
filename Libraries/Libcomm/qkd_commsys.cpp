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

#include "qkd_commsys.h"
#include <iostream>
#include <sstream>

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
// 2 - Stop when an error is introduced to a correctly-decoded frame
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

// Description & Serialization
template <class S, class T, template <class> class C>
std::string
qkd_commsys<S, T, C>::description() const
{
    std::ostringstream sout;
    sout << "QKD communication System: ";
    sout << alice_channel->description() << ", ";
    sout << bob_channel->description() << ", ";
    sout << protocol->description();
    // sout << cdc->description();
    return sout.str();
}

// object serialization - saving

template <class S, class T, template <class> class C>
std::ostream&
qkd_commsys<S, T, C>::serialize(std::ostream& sout) const
{
    // format version
    sout << "# Version" << std::endl;
    sout << 1 << std::endl;
    sout << "# Frame size (# of quantum states in a frame)" << std::endl;
    sout << framesize << std::endl;
    sout << "## Alice's channel" << std::endl;
    sout << alice_channel << std::endl;
    sout << "## Bob's channel" << std::endl;
    sout << bob_channel << std::endl;
    sout << "## Postprocessing protocol" << std::endl;
    sout << protocol;

    return sout;
}

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering; added split channel model
 */
template <class S, class T, template <class> class C>
std::istream&
qkd_commsys<S, T, C>::serialize(std::istream& sin)
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

    sin >> libbase::eatcomments >> framesize >> libbase::verify;
    sin >> libbase::eatcomments >> alice_channel >> libbase::verify;
    sin >> libbase::eatcomments >> bob_channel >> libbase::verify;
    sin >> libbase::eatcomments >> protocol >> libbase::verify;

    assertalways(sin.good());
    return sin;
}

} // namespace libcomm

// namespace libcomm
// {

// // Explicit Realizations
// // TO ADD MORE depending on protocol needed
// // E.g.
// // template qkd_commsys<qubit, bool>;

// // qkd_commsys<S, T, C>
// // Template class for the CV-QKD protocol (GG02) using Gaussian modulated
// coherent states template class qkd_commsys<gaussian_state, double,
// libbase::vector>; } // namespace libcomm

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
#define CONTAINER_SEQ (libbase::vector)

/* Serialization string (S, T, C):  qkd_commsys<gaussian_state, double,
 * libbase::vector> where: S = gaussian_state .. T = double, float C =
 * libbase::vector
 */

#define INSTANTIATE(r, args) \
    template class qkd_commsys<BOOST_PP_SEQ_ENUM(args)>; \
    template <> \
    const libbase::serializer qkd_commsys<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "qkd_commsys", \
            "qkd_commsys<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1, args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2, args)) ">", \
            qkd_commsys<BOOST_PP_SEQ_ENUM(args)>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(
    INSTANTIATE, (STATE_SEQ)(SCALAR_SEQ)(CONTAINER_SEQ))

} // namespace libcomm
