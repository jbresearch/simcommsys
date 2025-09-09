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

    /* Still to add (already implemented): randgen rng; and set seed of src and system? */
    libbase::vector<S> source = src->generate_sequence(sys->input_block_size());

    // k is known from codec of the cv-qkd protocol.
    int k = get_codec_input_bits_k();

    // Generate vector s for Bob with size k.
    vector_s = create_vector_s(k);

    /* Still to add (already implemented):

    int framesize = sys->input_block_size(); // Number of generated states per frame from Alice.

    // Setting modulation variance VA in the gaussian quantum channel of Bob
    double VA = src->get_VA();
    sys->set_VA(*src);
    */

    libbase::vector<bool> skey = sys->fullcycle(source);

    libbase::indirect_vector<double> result_segment =
        result.segment(0, R::count());

    /*
    // Still to add (already implemented):

    int l_secret_key = sys->protocol->get_length_secret_key();

    // Still to change update_results of results collector to:
    R::updateresults(result_segment, source, skey, l_secret_key, framesize);

    */

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

// To be used for the R (third) templated parameter.
#include "result_collector/qkd_commsys/cv_qkd_errors_hamming.h"
// Explicit Realizations
// TODO
// E.g.
// template qkd_commsys_simulator<qubit, bool>;

} // namespace libcomm
