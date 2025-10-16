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

#include "commsys_simulator.h"

#include "channel_insdel.h"
#include "commsys_stream.h"
#include "modem/stream_modulator.h"
#include "result_collector/commsys/fidelity_pos.h"
#include "source/sequential.h"
#include "source/uniform.h"
#include "source/zero.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
// 2 - For fidelity collector, observe actual/estimated boundary drifts
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

// *** Templated Common Base ***

// Experiment handling

/*!
 * \brief Perform a complete encode->transmit->receive cycle
 * \param[out] result   Vector containing the set of results to be updated
 *
 * Results are organized according to the collector used, as a function of
 * the iteration count.
 *
 * \note The results collector assumes that the result vector is an accumulator,
 * so that every call adds to the existing result. This explains the need to
 * initialize the result vector to zero.
 */
template <class S>
void
commsys_simulator<S>::sample(libbase::vector<double>& result)
{
    // Reset timers
    this->reset_timers();
    // Initialise result vector
    result.init(count());
    result = 0;
    // Get access to the results collector in codeword boundary analysis mode
    fidelity_pos* rc_fidelity = dynamic_cast<fidelity_pos*>(rc.get());

    // Create source stream
    libbase::vector<int> source =
        src->generate_sequence(sys->input_block_size());
#if DEBUG >= 2
    std::cout << "Source stream: " << source << std::endl;
#endif

    // Encode -> Map -> Modulate
    libbase::vector<S> transmitted = sys->encode_path(source);
#if DEBUG >= 2
    std::cout << "Transmitted: " << transmitted << std::endl;
#endif

    // Transmit
    libbase::vector<S> received = sys->transmit(transmitted);
#if DEBUG >= 2
    std::cout << "Received: " << received << std::endl;
#endif

    // Demodulate -> Inverse Map -> Translate
    sys->receive_path(received);

    // Decode
    if (analyze_decode_iters && !rc_fidelity) {
        // We check that rc since analyze_decode_iters does not matter in
        // context of codeword boundary analysis; we always need all iters

        // Collect codewords at each iteration
        libbase::vector<libbase::vector<int>> decoded;
        sys->decode(decoded);
#if DEBUG >= 2
        std::cout << "Decoded: " << decoded(this->num_iters() - 1) << std::endl;
#endif

        // collect results for each iteration
        for (int curr_cdc_iter = 0; curr_cdc_iter < this->sys->num_iter();
             curr_cdc_iter++) {
            libbase::indirect_vector<double> result_segment =
                result.segment(curr_cdc_iter * rc->count(), rc->count());
            rc->updateresults(result_segment, source, decoded(curr_cdc_iter));
        }

        // Keep record of what we last simulated
        const int tau = sys->input_block_size();
        assert(source.size() == tau);
        for (int curr_cdc_iter = 0; curr_cdc_iter < this->sys->num_iter();
             curr_cdc_iter++)
            assert(decoded(curr_cdc_iter).size() == tau);
        last_event.init(2 * tau);
        for (int i = 0; i < tau; i++) {
            last_event(i) = source(i);
            last_event(i + tau) = decoded(this->sys->num_iter() - 1)(i);
        }

    } else { // We collect results for last iteration only

        libbase::vector<int> decoded;
        sys->decode(decoded);
#if DEBUG >= 2
        std::cout << "Decoded: " << decoded << std::endl;
#endif

        if (!rc_fidelity) {
            libbase::indirect_vector<double> result_segment =
                result.segment(0, rc->count());
            rc->updateresults(result_segment, source, decoded);

        } else { // perform codeword boundary analysis if this is indicated

            // Get access to the modem in stream mode
            stream_modulator<S, libbase::vector>& modem_stream =
                dynamic_cast<stream_modulator<S, libbase::vector>&>(
                    *sys->getmodem());
            // Get access to the TX channel in insdel mode
            channel_insdel<S, real>& txchan_insdel =
                dynamic_cast<channel_insdel<S, real>&>(*sys->gettxchan());

            // get codeword boundary positions from modem (encoder-side)
            const array1i_t boundary_pos = modem_stream.get_boundaries();
            // get actual drift at codeword boundary positions from channel
            // (decoder-side)
            const array1i_t act_drift = txchan_insdel.get_drift(boundary_pos);

            // get estimated drift pdfs
            array1vd_t post_pdftable;
            libbase::size_type<libbase::vector> offset;
            modem_stream.get_post_drift_pdf(post_pdftable, offset);
            // get most probable estimated drift positions
            array1i_t est_drift(post_pdftable.size());
            for (int i = 0; i < post_pdftable.size(); i++) {
                est_drift(i) =
                    commsys_stream<S, libbase::vector, real>::estimate_drift(
                        post_pdftable(i), offset);
            }

            // Tell user what we're doing
#if DEBUG >= 4
            std::cerr << "DEBUG (commsys_simulator): act bdry drift = "
                      << act_drift << std::endl;
            std::cerr << "DEBUG (commsys_simulator): est bdry drift = "
                      << est_drift << std::endl;
#endif
            // accumulate results
            rc->updateresults(result, act_drift, est_drift);
        }

        // Keep record of what we last simulated
        const int tau = sys->input_block_size();
        assert(source.size() == tau);
        assert(decoded.size() == tau);
        last_event.init(2 * tau);
        for (int i = 0; i < tau; i++) {
            last_event(i) = source(i);
            last_event(i + tau) = decoded(i);
        }
    }
}

// Description & Serialization

template <class S>
std::string
commsys_simulator<S>::description() const
{
    std::ostringstream sout;
    sout << "Simulator for ";
    sout << sys->description();
    sout << ", ";
    sout << src->description();
    sout << ", collecting ";
    sout << rc->description();
    return sout.str();
}

// object serialization - saving

template <class S>
std::ostream&
commsys_simulator<S>::serialize(std::ostream& sout) const
{
    // format version
    sout << "# Version" << std::endl;
    sout << 5 << std::endl;
    sout << "# Analyze all decode iterations" << std::endl;
    sout << analyze_decode_iters << std::endl;
    sout << "# Source generator" << std::endl;
    sout << src;
    sout << "# Communication system" << std::endl;
    sout << sys;
    sout << "# Results collector" << std::endl;
    sout << rc;
    return sout;
}

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added input mode parameter and support for all-zero input
 *
 * \version 2 Added support for user-supplied sequence of input symbols
 *
 * \version 3 Using source-generator object
 *
 * \version 4 Adding option to analyze all decode iterations.
 *
 * \version 5 Adding results collector
 */

template <class S>
std::istream&
commsys_simulator<S>::serialize(std::istream& sin)
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
    // get analyze_decode_iters if version is right
    if (version >= 4) {
        sin >> libbase::eatcomments >> this->analyze_decode_iters >>
            libbase::verify;
    } else {
        // default is to analyze decode iters, to preserve backwards
        // compatibility.
        this->analyze_decode_iters = true;
    }
    // source-generator section depending on version
    if (version >= 3) {
        // source generator
        sin >> libbase::eatcomments >> src >> libbase::verify;
        assertalways(src);
    } else if (version >= 1) {
        int temp;
        // read input mode
        sin >> libbase::eatcomments >> temp >> libbase::verify;
        // interpret mode
        enum input_mode_t {
            input_mode_zero = 0,        //!< All-zero input
            input_mode_random,          //!< Random input
            input_mode_user_sequential, //!< Sequentially-applied user sequence
            input_mode_undefined
        };
        assertalways(temp >= 0 && temp < input_mode_undefined);
        input_mode_t input_mode = static_cast<input_mode_t>(temp);
        switch (input_mode) {
        case input_mode_zero:
            // create source generator
            src.reset(new zero<int>);
            break;

        case input_mode_random:
            // do this later (needs alphabet size from system)
            src.reset();
            break;

        case input_mode_user_sequential: {
            // read count of input symbols
            sin >> libbase::eatcomments >> temp >> libbase::verify;
            // read input symbols from stream
            array1i_t input_vectors(temp);
            sin >> libbase::eatcomments;
            input_vectors.serialize(sin);
            libbase::verify(sin);
            // create source generator
            src.reset(new sequential<int>(input_vectors));
        } break;

        default:
            failwith("Unknown input mode");
            break;
        }
    } else {
        // do this later (needs alphabet size from system)
        src.reset();
    }

    // communication system object
    sin >> libbase::eatcomments >> sys >> libbase::verify;
    assertalways(sys);

    // get results collector if version is right
    if (version >= 5) {
        sin >> libbase::eatcomments >> rc >> libbase::verify;
    } else {
        failwith("Results collector not specified");
    }

    // create source generator if not done yet
    if (!src) {
        src.reset(new uniform<int>(sys->num_inputs()));
    }

    // initialise components
    rc->init(*this);

    // finish
    assertalways(sin.good());
    return sin;
}

} // namespace libcomm

#include "erasable.h"
#include "gf.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::erasable;
using libbase::serializer;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define FINITE_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

#define ADD_ERASABLE(r, x, type) \
   (type)(erasable<type>)

#define ALL_FINITE_TYPE_SEQ \
   BOOST_PP_SEQ_FOR_EACH(ADD_ERASABLE, x, FINITE_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace) \
   ALL_FINITE_TYPE_SEQ

/* Serialization string: commsys_simulator<type>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
      template class commsys_simulator<type>; \
      template <> \
      const serializer commsys_simulator<type>::shelper( \
            "experiment", \
            "commsys_simulator<" BOOST_PP_STRINGIZE(type) ">", \
            commsys_simulator<type>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // namespace libcomm
