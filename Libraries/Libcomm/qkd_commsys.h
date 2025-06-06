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
class qkd_commsys : public instrumented,
                    public parametric,
                    public libbase::serializable
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

    //! \brief How many quantum states in one frame
    int framesize;
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

    /*! \name Parametric interface */
    void set_parameters(const libbase::vector<double>& x) override
    {
        assertalways(x.size() == this->get_num_params());

        libbase::vector<double> alice_channel_params;
        alice_channel_params.init(this->alice_channel->get_num_params());
        int i = 0;
        for (; i < this->alice_channel->get_num_params(); i++) {
            alice_channel_params(i) = x(i);
        }

        libbase::vector<double> bob_channel_params;
        bob_channel_params.init(this->bob_channel->get_num_params());
        int j = 0;
        for (; j < this->bob_channel->get_num_params(); i++, j++) {
            bob_channel_params(j) = x(i);
        }

        this->alice_channel->set_parameters(alice_channel_params);
        this->bob_channel->set_parameters(bob_channel_params);
    }
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(get_num_params());

        libbase::vector<double> alice_channel_params =
            this->alice_channel->get_parameters();
        int i = 0;
        for (; i < this->alice_channel->get_num_params(); i++) {
            params(i) = alice_channel_params(i);
        }

        libbase::vector<double> bob_channel_params = this->bob_channel->get_parameters();
        int k = i; // continue from where alice_channel_params left off
        for (int j = 0; j < this->bob_channel->get_num_params(); j++, k++) {
            params(k) = bob_channel_params(j);
        }

        return params;
    }
    int get_num_params() const override
    {
        return this->alice_channel->get_num_params() +
               this->bob_channel->get_num_params();
    }
    // @}

    /*! \name Communication System Interface */
    //! Perform complete transmission of one frame
    C<bool> fullcycle(const C<S>& source)
    {
        assertalways(source.size() == framesize);

        libbase::vector<std::unique_ptr<observable<T>>> alice_observables =
            protocol->get_alice_observables(framesize);
        libbase::vector<std::unique_ptr<observable<T>>> bob_observables =
            protocol->get_bob_observables(framesize);

        // create and allocate vectors for measurements on Bob and Alice's end
        libbase::vector<T> alice_measurements;
        libbase::vector<T> bob_measurements;

        alice_measurements.init(framesize);
        bob_measurements.init(framesize);

        for (int i = 0; i < framesize; i++) {
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

        return protocol->postprocess(alice_measurements, bob_measurements);
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

    //! Get number of input quantum states in a frame
    int input_block_size() const { return framesize; }

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_BASE_SERIALIZER(qkd_commsys)
    DECLARE_SERIALIZER(qkd_commsys)
};

} // namespace libcomm

#endif // __qkd_commsys_h