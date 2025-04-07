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

#ifndef __qkd_commsys_h
#define __qkd_commsys_h

#include "commsys.h"

#include "qkd/observable.h"
#include "qkd/qkd_protocol.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"
#include "serializer.h"
#include "vector.h"

#include <iostream>
#include <memory>
#include <sstream>

namespace libcomm
{

/*!
 * \brief   Common Base for QKD System.
 * \author  Mark Mizzi
 */

template <class S, class T, template <class> class C = libbase::vector>
class qkd_commsys : public instrumented, public libbase::serializable
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<double> array1d_t;
    // @}

protected:
    /*! \name Bound objects */
    std::unique_ptr<quantum_channel> bob_channel;
    std::unique_ptr<quantum_channel> alice_channel;
    std::unique_ptr<qkd_protocol<T, C>> protocol;
    // @}
public:
    qkd_commsys() {}
    // @}

    /*! \name Communication System Setup */
    void seedfrom(libbase::random& r)
    {
        this->alice_channel->seedfrom(r);
        this->bob_channel->seedfrom(r);
        this->protocol->seedfrom(r);
    }
    // @}

    /*! \name Communication System Interface */
    //! Perform complete transmission of one frame
    C<bool> fullcycle(const C<S>& source)
    {
        libbase::vector<std::unique_ptr<observable<T>>> alice_observables =
            protocol->get_alice_observables(source.size());
        libbase::vector<std::unique_ptr<observable<T>>> bob_observables =
            protocol->get_bob_observables(source.size());

        // create and allocate vectors for measurements on Bob and Alice's end
        libbase::vector<T> alice_measurements;
        libbase::vector<T> bob_measurements;

        alice_measurements.init(source.size());
        bob_measurements.init(source.size());

        for (int i = 0; i < source.size(); i++) {
            // Quantum channel transmission
            alice_observables(i)->transmit(*this->alice_channel);
            bob_observables(i)->transmit(*this->bob_channel);

            // Measurement of quantum states
            if (S::is_entangled) {
                alice_measurements(i) =
                    source(i).measure(*alice_observables(i), 0);
                bob_measurements(i) = source(i).measure(*bob_observables(i), 1);
            } else {
                alice_measurements(i) =
                    source(i).measure(*alice_observables(i));
                bob_measurements(i) = source(i).measure(*bob_observables(i));
            }
        }

        protocol->postprocess(alice_measurements, bob_measurements);
    }
    // @}

    //! Clear list of timers
    void reset_timers() override
    {
        // clear list of timers we're keeping
        instrumented::reset_timers();
        // clear list of timers for all components
        protocol->reset_timers();
    }

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_BASE_SERIALIZER(qkd_commsys)
    DECLARE_SERIALIZER(qkd_commsys)
};

} // namespace libcomm

#endif // __qkd_commsys_h