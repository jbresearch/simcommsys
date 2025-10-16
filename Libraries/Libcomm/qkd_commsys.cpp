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

/*! \name Communication System Interface */
//! Perform complete transmission of one frame
template <class S, class T, template <class> class C>
std::pair<C<bool>, C<bool>>
qkd_commsys<S, T, C>::fullcycle(C<S>& source)
{
    // ***** Note: In this case the source here is the libbase::vector of
    // states e.g. coherent states if S=gaussian_State *****
    assertalways(source.size() == framesize);

    // Note: Here I Changed the libbase::vector to an std::vector only for
    // the observables stage
    std::vector<std::unique_ptr<observable<T>>> bob_observables =
        protocol->get_bob_observables(framesize);

    std::vector<std::unique_ptr<observable<T>>> alice_observables =
        protocol->get_alice_observables(framesize);

    // Create and allocate vectors for measurements on Bob and Alice's end
    libbase::vector<T> alice_measurements;
    libbase::vector<T> bob_measurements;

    alice_measurements.init(framesize);
    bob_measurements.init(framesize);

    for (int i = 0; i < framesize; i++) {
        // Quantum channel transmission
        alice_observables[i]->transmit(*this->alice_channel);
        bob_observables[i]->transmit(*this->bob_channel);

        // Measurement of quantum states
        if constexpr (S::is_entangled) {
            alice_measurements(i) = source(i).measure(*alice_observables[i], 0);
            bob_measurements(i) = source(i).measure(*bob_observables[i], 1);
        } else {
            alice_measurements(i) = source(i).measure(*alice_observables[i]);
            bob_measurements(i) = source(i).measure(*bob_observables[i]);
        }
    }
    // Pass source generator to get_VA for CV-QKD and initialises Bob's
    // quantum channel.
    protocol->init(*get_src(), bob_channel);

    // Perform post-processing to get the final secret keys.
    auto [secret_key_KA, secret_key_KB] = protocol->postprocess(
        std::move(alice_measurements), std::move(bob_measurements));

    return {std::move(secret_key_KA), std::move(secret_key_KB)};
}
// @}

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

// Explicit Realizations
// TO ADD MORE depending on protocol needed
// E.g.
// template qkd_commsys<qubit, bool>;

namespace libcomm
{

// ----- explicit instantiations & serializer registration (Boost PP) -----
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::vector;

// clang-format off
#define STATE_SEQ (gaussian_state)
#define SCALAR_SEQ (double)
#define CONTAINER_SEQ (vector)

/* Serialization string (S, T, C); qkd_commsys<S, T, C> : qkd_commsys<gaussian_state, double,
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

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
                              (STATE_SEQ)(SCALAR_SEQ)(CONTAINER_SEQ))

} // namespace libcomm
