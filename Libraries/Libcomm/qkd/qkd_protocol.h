/*!
 * \file
 *qwqw
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

#ifndef __qkd_protocol_h
#define __qkd_protocol_h

#include "instrumented.h"
#include "qkd/observable.h"
#include "random.h"
#include "serializer.h"
#include "source.h"
#include "source.h"
#include "vector.h"
#include <memory>
#include <string>
#include <vector>

#include <type_traits>

namespace libcomm
{


/* Template Definition:
S= type of state e.g. gaussian_state, qubit ...
For S check quantum_state for all the types of states.
T = double, bool
C = libbase::vector
*/

// Forward Declaration
template <class S, class T, template <class> class C>
class qkd_commsys;

/*!
 * \brief   Common Base for QKD postprocessing protocol.
 * \author  Mark Mizzi, Aaron Abela, Aaron Abela
 */
template <class S, typename T, template <class> class C = libbase::vector>
class qkd_protocol : public instrumented, public libbase::serializable
{
public:

    // Allows the protocol to access anything from the qkd_commsys object.
    virtual void init(qkd_commsys<S, T, C>* qkdcommsys) = 0;

    // Pass the generated source sequence to the protocol at the start of a cycle.
    virtual void set_source_sequence(const C<S>& /* source_sequence */)
    {
        // Default implementation does nothing.
    }

    /*! Get observables used to measure quantum states, e.g. spin
     * in two different bases for E91
     * Integer param determines number of observables returned.
     *
     * Note: Changed only the observables to work with a std::vector rather than
     * a libbase::vector
     */
    virtual std::vector<std::unique_ptr<observable<T>>>
    get_alice_observables(int) = 0;

    virtual std::vector<std::unique_ptr<observable<T>>>
    get_bob_observables(int) = 0;

    // Split fn to be used for parameter estimation and post-processing.
    virtual std::tuple<libbase::vector<T>,
                       libbase::vector<T>,
                       libbase::vector<T>,
                       libbase::vector<T>>
    split(libbase::vector<T>& measurements_alice,
          libbase::vector<T>& measurements_bob) = 0;

    // Pass source generator to get VA for CV_QKD.

    /* Helper functions related to codec.*/
    virtual std::shared_ptr<codec<libbase::vector>> get_codec() const = 0;

    virtual int get_codec_input_bits_k() const = 0;
    virtual int get_codec_output_bits_n() const = 0;

    // Returns final secret keys KA and KB.
    virtual std::pair<C<bool>, C<bool>>
    postprocess(libbase::vector<T>&& alice_measurements,
                libbase::vector<T>&& bob_measurements) = 0;

    virtual void seedfrom(libbase::random& r) = 0;

    virtual std::string description() const = 0;

    virtual ~qkd_protocol() {}

    DECLARE_BASE_SERIALIZER(qkd_protocol)
};

} // end namespace libcomm

#endif // __qkd_protocol_h