/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi, Aaron Abela
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
#include "source.h"
#include "source.h"
#include "source/quantum_gaussian_source.h"
#include "source/quantum_bb84_source.h"
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
    std::shared_ptr<quantum_channel> bob_channel;
    std::shared_ptr<quantum_channel> alice_channel;
    std::shared_ptr<qkd_protocol<S, T, C>> protocol;
    std::shared_ptr<source<S>> src;

    //! \brief How many quantum states in one frame
    int framesize = 0;

    // @}
public:
    qkd_commsys() {}
    // @}

    // init method for qkd_commsys
    template <class Simulator>
    void init(Simulator* sim)
    {
        // Get source generator member of qkd_commsys
        this->src = sim->get_src_gen();

        assert(this->src && "Source generator not found in simulator.");
        assert(this->protocol && "Protocol has not been set.");

        // Call the protocol's init and pass the qkd_commsys 'this' pointer/ the qkd_commsys object.
        this->protocol->init(this);
    }


    /*! \name Communication System Setup */
    void seedfrom(libbase::random& r)
    {
        this->alice_channel->seedfrom(r);
        this->bob_channel->seedfrom(r);
        this->protocol->seedfrom(r);
        observable<T>::seedfrom(r);
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

        libbase::vector<double> bob_channel_params =
            this->bob_channel->get_parameters();
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

    // Getter for Bob's channel
    std::shared_ptr<quantum_channel> get_bob_channel() { return bob_channel; }

    // Gets the source generator from qkd_commsys_simulator
    std::shared_ptr<source<S>> get_src() { return this->src; }

    // Getter to get the input bits from the codec from cvqkd_protocol.h.
    int get_codec_input_bits_k() { return protocol->get_codec_input_bits_k(); }

    /*! \name Communication System Interface */
    std::pair<C<bool>, C<bool>> fullcycle(C<S>& source);

    //! Clear list of timers
    void reset_timers()
    {
        // clear list of timers we're keeping
        instrumented::reset_timers();
        // clear list of timers for all components
        protocol->reset_timers();
    }

    //! Get number of input quantum states in a frame.
    int input_block_size() const { return framesize; }

    // Description
    std::string description() const;

    // Serialization Support using shared pointers
    DECLARE_BASE_SERIALIZER(qkd_commsys)
    DECLARE_SERIALIZER(qkd_commsys)
};

} // namespace libcomm

#endif // __qkd_commsys_h