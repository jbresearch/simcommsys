/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi
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

    libbase::vector<S> source = src->generate_sequence(sys->input_block_size());
    libbase::vector<bool> skey = sys->fullcycle(source);

    libbase::indirect_vector<double> result_segment =
        result.segment(0, R::count());
    R::updateresults(result_segment, source, skey);
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

namespace libcomm
{

// Explicit Realizations
// TODO
// E.g.
// template qkd_commsys_simulator<qubit, bool>;

} // namespace libcomm
