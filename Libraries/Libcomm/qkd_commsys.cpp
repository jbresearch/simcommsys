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

// // Clone (deep copy via serialization)
// template <class S, class T, template <class> class C>
// std::shared_ptr<libbase::serializable>
// qkd_commsys<S, T, C>::clone() const
// {
//     // Note: Avoids copy-constructing unique_ptr members by round-tripping
//     // through the serializer.
//     auto out = std::make_shared<qkd_commsys<S, T, C>>();

//     std::stringstream ss;
//     this->serialize(ss);  // write "this" into the stream
//     out->serialize(ss);   // read into the new object

//     return out;
// }

} // namespace libcomm

// namespace libcomm
// {

// // Explicit Realizations
// // TO ADD MORE depending on protocol needed
// // E.g.
// // template qkd_commsys<qubit, bool>;

// // qkd_commsys<S, T, C>
// // Template class for the CV-QKD protocol (GG02) using Gaussian modulated coherent states
// template class qkd_commsys<gaussian_state, double, libbase::vector>;
// } // namespace libcomm

// ----- explicit instantiations & serializer registration (Boost PP) -----
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/stringize.hpp>

namespace libcomm
{

// define the type sequences you want to build (extend as you add support)
#define QKD_STATE_SEQ     (gaussian_state)      /* add more states here */
#define QKD_SCALAR_SEQ    (double)                       /* e.g. (double)(float) */
#define QKD_CONTAINER_SEQ (libbase::vector)              /* e.g. (libbase::vector)(libbase::matrix) */

/* For each cartesian product (S, T, C):
 *  - explicitly instantiate the template class
 *  - define the serializer helper singleton (shelper) like commsys.cpp
 */
#define QKD_INSTANTIATE(r, args)                                                              \
    template class qkd_commsys<BOOST_PP_SEQ_ENUM(args)>;                                      \
    template<>                                                                                \
    const libbase::serializer                                                                 \
    qkd_commsys<                                                                              \
        BOOST_PP_SEQ_ELEM(0, args), /* S */                                                   \
        BOOST_PP_SEQ_ELEM(1, args), /* T */                                                   \
        BOOST_PP_SEQ_ELEM(2, args)  /* C */                                                   \
    >::shelper(                                                                               \
        "qkd_commsys",                                                                        \
        "qkd_commsys<"                                                                        \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, args)) ","                                \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1, args)) ","                                \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2, args)) ">",                                \
        qkd_commsys<BOOST_PP_SEQ_ENUM(args)>::create                                          \
    );

BOOST_PP_SEQ_FOR_EACH_PRODUCT(QKD_INSTANTIATE,
                              (QKD_STATE_SEQ)(QKD_SCALAR_SEQ)(QKD_CONTAINER_SEQ))

} // namespace libcomm
