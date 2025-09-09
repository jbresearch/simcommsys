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

    // Check that verifies if I_AB > X_BE?
    int MI_check = 0;

    libbase::vector<bool> vector_s_from_bob;
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

    // Getter to get the input bits from the codec from cvqkd_protocol.h.
    int get_codec_input_bits_k()
    {
        return protocol->get_codec_input_bits_k();
    }

    // Setter method to get vector s which is generated in the qkd_commsys_simulator.
    void set_bob_vector(const libbase::vector<bool>& vector_s)
    {
        vector_s_from_bob = vector_s;
    }

    /*! \name Communication System Interface */
    //! Perform complete transmission of one frame
    C<bool> fullcycle(C<S>& source)
    {
        // ***** Note: In this case the source here is the libbase::vector of states e.g. coherent states if S=gaussian_State *****
        assertalways(source.size() == framesize);

        // Note: Here I Changed the libbase::vector to an std::vector only for the observables stage
        std::vector<std::unique_ptr<observable<T>>> bob_observables = protocol->get_bob_observables(framesize);

        // Get Bob's decision vector of his observables.
        const libbase::vector<int>& decision_vector = protocol->get_decision_vector();

        std::vector<std::unique_ptr<observable<T>>> alice_observables =
            protocol->get_alice_observables(framesize, decision_vector);

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
                alice_measurements(i) =
                    source(i).measure(*alice_observables[i], 0);
                bob_measurements(i) = source(i).measure(*bob_observables[i], 1);
            } else {
                alice_measurements(i) =
                    source(i).measure(*alice_observables[i]);
                bob_measurements(i) = source(i).measure(*bob_observables[i]);
            }
        }

        /* Perform Parameter Estimation*/
        // Required parameters for parameter estimation.
        int N_PE = protocol->get_N_PE();
        std::cout << "\n (prints from qkd_commsys.h) Number of States used for Parameter Estimation = " << N_PE << std::endl;
        int N_0 = protocol->get_N_0();
        double v_el = protocol->get_v_el();
        double detector_efficiency = protocol->get_det_eff();

        // Perform split for parameter estimation and post-processing.
        auto [X_PE, Y_PE, X_raw, Y_raw] = protocol->split(alice_measurements, bob_measurements, N_PE);

        // Calculate parameter estimation using optical fiber.
        auto [T_hat, Epsilon_hat, chi_total_hat] = protocol->parameter_estimation_optical_fiber(X_PE, Y_PE, N_0, v_el, detector_efficiency);

        double modulation_variance = this->bob_channel->get_VA();
        double V = modulation_variance + 1;

        /* Calculate Mutual Information. */
        double I_AB = protocol->calculate_mutual_information(chi_total_hat, modulation_variance);
        std::cout << "(Prints from qkd_commsys.h) I_AB = " << I_AB << std::endl;

         /* Calculate Holevo Bound. */
        double X_BE = protocol->calculate_holevo_bound(V, T_hat, Epsilon_hat, chi_total_hat);
        std::cout << "(Prints from qkd_commsys.h) X_BE = " << X_BE << std::endl;

        // Checks whether the protocol is aborted or not to continue with the Information Reconciliation stage.
        if(I_AB > X_BE)
        {
            MI_check = 1;
            std::cout << "(Prints from qkd_commsys.h) MI_Check = " << MI_check << std::endl; // To delete

            double n_samples = 99000; // still to get from codec. Length of codeword.
            double beta_mdr = 0.958;
            std::cout << "(prints from qkd_commsys.h full cycle) beta_mdr = " << beta_mdr << std::endl;

            /* Gets SNR_linear from Bob's Gaussian Quantum Channel -> to uncomment!!
            libbase::vector<double> bobs_channel_parameters;
            bobs_channel_pararmeters.init(1);
            bobs_channel_parameters= bob_channel->get_parameters();
            double SNR_linear = bobs_channel_parameters(0);
            */

            // *** Will also need to use the SNR_linear later to  get SNR_dB to use it in the codec for decoding in the post processing probably. ***/

            /* Confirmed: Beta equation that needs to be used: \beta = R/C(S) taken from the Quasi Cyclic Paper 2018, Mario Milicevic.
            // calculate as I still need R from codec and SNR from channel.
            double code_rate = 0.02;
            double beta_mdr = code_rate/(protocol->calculate_shannon_capacity_awgn(SNR_linear)); // still to Beta is coming too small
            */

            // Use setter in CV-QKD protocol
            protocol->set_parameters_secret_key_length(beta_mdr, I_AB, X_BE, n_samples);

            // Continue with post-processing: Still to implement
            return protocol->postprocess(std::move(X_raw), std::move(Y_raw));
            // std::move was required due to the following: Was passing lvalues to a function that expects rvalue references (&&).
            // return protocol->postprocess(alice_measurements, bob_measurements); // To check with Mark why in the qkd_protocol.h for the post-processing method he used &&?
        }
        else
        {
            MI_check = 0;
            int len_secret_key = 0;

            libbase::vector<bool> final_secret_key;
            final_secret_key.init(len_secret_key);
            return final_secret_key; // Returns an empty libbase bool vector.
        }

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

    //! Get number of input quantum states in a frame.
    int input_block_size() const { return framesize; }

    // Description
    std::string description() const;

    // Getter to get the protocol to be used in the TestGaussiancvqkd. To remove when using simulator.
    auto get_protocol()
    {
        return protocol;
    }

    // Serialization Support using shared pointers
    DECLARE_BASE_SERIALIZER(qkd_commsys)
    DECLARE_SERIALIZER(qkd_commsys)
};

} // namespace libcomm

#endif // __qkd_commsys_h