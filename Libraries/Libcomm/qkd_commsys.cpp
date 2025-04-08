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
    sout << "QKD ommunication System: ";
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
} // namespace libcomm

namespace libcomm
{

// Explicit Realizations
// TODO

} // namespace libcomm
