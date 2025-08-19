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
#include "source/quantum_gaussian_source.h"
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
    // std::unique_ptr<quantum_channel> bob_channel;
    // std::unique_ptr<quantum_channel> alice_channel;
    // std::unique_ptr<qkd_protocol<T, C>> protocol;
    std::shared_ptr<quantum_channel> bob_channel;
    std::shared_ptr<quantum_channel> alice_channel;
    std::shared_ptr<qkd_protocol<T, C>> protocol;

    //! \brief How many quantum states in one frame
    int framesize = 0;
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

    // Sets the modulation variance VA to be used by the gaussian quantum channel to calculate variance VN from it.
    void set_VA(libcomm::quantum_gaussian_source& source)
    {
        double VA = source.get_VA();
        this->bob_channel->set_VA(VA);
    }

    /*! \name Communication System Interface */
    //! Perform complete transmission of one frame
    C<bool> fullcycle(C<S>& source)
    {
        // ***** Note: In this case the source here is the libbase::vector of states e.g. coherent states if S=gaussian_State *****

        assertalways(source.size() == framesize);

        // Note: Here I Changed the libbase::vector to an std::vector only for the observables stage
        std::vector<std::unique_ptr<observable<T>>> alice_observables =
            protocol->get_alice_observables(framesize);
        std::vector<std::unique_ptr<observable<T>>> bob_observables =
            protocol->get_bob_observables(framesize);

        // create and allocate vectors for measurements on Bob and Alice's end
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
                alice_measurements(i) =
                    source(i).measure(*alice_observables[i], 0);
                bob_measurements(i) = source(i).measure(*bob_observables[i], 1);
            } else {
                alice_measurements(i) =
                    source(i).measure(*alice_observables[i]);
                bob_measurements(i) = source(i).measure(*bob_observables[i]);
            }
        }

        // return protocol->postprocess(alice_measurements, bob_measurements)
        return protocol->postprocess(std::move(alice_measurements), std::move(bob_measurements)); // To check with Mark why in the qkd_protocol.h for the post-processing method he used &&?
    }
    // @}

    /*! \name Communication System Interface */
    //! Perform complete transmission of one frame specifically using a libcomm::quantum_gaussian_source

    // ***** Note VIMP: This will have to change back to a sequence as done in the original fullcycle, set_VA will have to be called in the simulator AND the original source will also be called in the simulator to create the sequence of coherent states. Then that sequence is the input to the full cycle method. For now I am just using fullcycle like this to test up until measurement.

    // C<bool> fullcycle(libcomm::quantum_gaussian_source& source)
    std::pair<libbase::vector<T>, libbase::vector<T>> fullcycle(libcomm::quantum_gaussian_source& source)
    {

        // Note: Here I Changed the libbase::vector to an std::vector only for the observables stage
        std::vector<std::unique_ptr<observable<T>>> bob_observables =
            protocol->get_bob_observables(framesize);

        // Get Bob's decision vector of his observables
        const libbase::vector<int>& decision_vector = protocol->get_decision_vector();

        std::vector<std::unique_ptr<observable<T>>> alice_observables =
        protocol->get_alice_observables(framesize, decision_vector);

        // create and allocate vectors for measurements on Bob and Alice's end
        libbase::vector<T> alice_measurements;
        libbase::vector<T> bob_measurements;

        alice_measurements.init(framesize);
        bob_measurements.init(framesize);

        // Setting modulation variance VA in the gaussian quantum of Bob
        set_VA(source);

        // Vector of coherent states where S = gaussian_state
        libbase::vector<S> source_sequence = source.generate_sequence(libbase::size_type<libbase::vector>(framesize));

        assertalways(source_sequence.size() == framesize);

        for (int i = 0; i < framesize; i++) {
            // Quantum channel transmission
            alice_observables[i]->transmit(*this->alice_channel);
            bob_observables[i]->transmit(*this->bob_channel);

            // Measurement of quantum states
            if constexpr (S::is_entangled) {
                alice_measurements(i) =
                    source_sequence(i).measure(*alice_observables[i], 0);
                bob_measurements(i) = source_sequence(i).measure(*bob_observables[i], 1);
            } else {
                alice_measurements(i) =
                    source_sequence(i).measure(*alice_observables[i]);
                bob_measurements(i) = source_sequence(i).measure(*bob_observables[i]);
            }
        }

        // return protocol->postprocess(alice_measurements, bob_measurements);
        return { std::move(alice_measurements), std::move(bob_measurements) }; // Just to test pre-processing.
    }
    // @}


    //! Clear list of timers
    void reset_timers()
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

    // Serialization Support using shared pointers
    DECLARE_BASE_SERIALIZER(qkd_commsys)
    DECLARE_SERIALIZER(qkd_commsys)

    // libbase::serializable interface using unique pointers
    // const std::string name() const override { return "qkd_commsys"; }
    // std::ostream& serialize(std::ostream& sout) const override;
    // std::istream& serialize(std::istream& sin) override;
    // std::shared_ptr<libbase::serializable> clone() const override; //override the clone() found in libbase::serialize

};

} // namespace libcomm

#endif // __qkd_commsys_h