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
#include "source/quantum_gaussian_source.h"
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

    // Sets the modulation variance VA to be used by the gaussian quantum
    // channel to calculate variance VN from it.
    void set_VA(libcomm::quantum_gaussian_source& source)
    {
        double VA = source.get_VA();
        this->bob_channel->set_VA(VA);
    }

    // Getter to get the input bits from the codec from cvqkd_protocol.h.
    int get_codec_input_bits_k() { return protocol->get_codec_input_bits_k(); }

    // Setter method to get vector s which is generated in the
    // qkd_commsys_simulator.
    void set_bob_vector(const libbase::vector<bool>& vector_s)
    {
        vector_s_from_bob = vector_s;
    }

    /*! \name Communication System Interface */
    //! Perform complete transmission of one frame
    std::pair<C<bool>, C<bool>> fullcycle(C<S>& source)
    {
        // ***** Note: In this case the source here is the libbase::vector of
        // states e.g. coherent states if S=gaussian_State *****
        assertalways(source.size() == framesize);

        // Note: Here I Changed the libbase::vector to an std::vector only for
        // the observables stage
        std::vector<std::unique_ptr<observable<T>>> bob_observables =
            protocol->get_bob_observables(framesize);

        // Get Bob's decision vector of his observables.
        const libbase::vector<int>& decision_vector =
            protocol->get_decision_vector();

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
        int n_output_codec_block_size = protocol->get_codec_output_bits_n();

        // Number of samples used for Parameter Estimation
        // N_PE = N (number of generated states) - n (size of codeword of the
        // codec)
        int N_PE = framesize - n_output_codec_block_size;

        std::cout << "\n (prints from qkd_commsys.h) Number of States used for "
                     "Parameter Estimation = "
                  << N_PE << std::endl;

        // N_0, v_el and detector efficiency are all serialized parameters in
        // the cv-qkd protocol. They are set in the configuration file.
        int N_0 = protocol->get_N_0();
        double v_el = protocol->get_v_el();
        double detector_efficiency = protocol->get_det_eff();

        // Perform split for parameter estimation and post-processing.
        auto [X_PE, Y_PE, X_raw, Y_raw] =
            protocol->split(alice_measurements, bob_measurements, N_PE);

        std::cout << "\n (prints from qkd_commsys.h) Size of X_PE and Y_PE: "
                  << X_PE.size() << "\t" << Y_PE.size() << std::endl;

        std::cout << "\n (prints from qkd_commsys.h) Size of X_Raw and Y_Raw: "
                  << X_PE.size() << "\t" << Y_PE.size() << std::endl;

        // Calculate parameter estimation using optical fiber.
        auto [T_hat, Epsilon_hat, chi_total_hat] =
            protocol->parameter_estimation_optical_fiber(
                X_PE, Y_PE, N_0, v_el, detector_efficiency);

        double modulation_variance = this->bob_channel->get_VA();
        double V = modulation_variance + 1;

        /* Calculate Mutual Information. */
        double I_AB = protocol->calculate_mutual_information(
            chi_total_hat, modulation_variance);
        std::cout << "(Prints from qkd_commsys.h) I_AB = " << I_AB << std::endl;

        /* Calculate Holevo Bound. */
        double X_BE = protocol->calculate_holevo_bound(
            V, T_hat, Epsilon_hat, chi_total_hat);
        std::cout << "(Prints from qkd_commsys.h) X_BE = " << X_BE << std::endl;

        /* Vector s is first generated in qkd_commsys_simulator sample() method.
         * It is then also set in the qkd_commsys_simulator sample() to the
         * qkd_commsys object; so that then it is set in the cv-qkd protocol. */
        protocol->set_bob_vector_s(vector_s_from_bob);

        /* TODO: TO delete lines 236 and 237. I am just doing this for debugging
         * purposes since the framesize I started with was small/ */
        // I_AB = 1.05;
        // X_BE = 0.82;

        // Checks whether the protocol is aborted or not to continue with the
        // Information Reconciliation stage.
        if (I_AB > X_BE) {
            MI_check = 1;
            std::cout << "(Prints from qkd_commsys.h) MI_Check = " << MI_check
                      << std::endl; // To delete

            // // TODO: Still to calculate using beta = R/C(S)
            // // C(S) is the Shannon Capacity of an AWGN channel.
            // double beta_mdr = 0.958;
            // std::cout << "(prints from qkd_commsys.h full cycle) beta_mdr = "
            // << beta_mdr << std::endl;

            /* Gets SNR_linear from Bob's Gaussian Quantum Channel*/
            libbase::vector<double> bobs_channel_parameters;
            bobs_channel_parameters.init(1);
            bobs_channel_parameters = bob_channel->get_parameters();
            double SNR_linear = bobs_channel_parameters(0);

            // Set SNR_linear to be used in the CV-QKD protocol.
            protocol->set_SNR_linear(SNR_linear);

            // Use setter in CV-QKD protocol
            protocol->set_parameters_secret_key_length(
                I_AB, X_BE, n_output_codec_block_size);

            // Continue with post-processing.

            auto [secret_key_KA, secret_key_KB] =
                protocol->postprocess(std::move(X_raw), std::move(Y_raw));

            return {std::move(secret_key_KA), std::move(secret_key_KB)};

            // std::move was required due to the following: Was passing lvalues
            // to a function that expects rvalue references (&&). return
            // protocol->postprocess(alice_measurements, bob_measurements); //
            // To check with Mark why in the qkd_protocol.h for the
            // post-processing method he used &&?
        } else {
            MI_check = 0;
            std::cout << "(Prints from qkd_commsys.h) MI_Check = " << MI_check
                      << std::endl; // To delete

            int len_secret_key = 0;

            libbase::vector<bool> secret_key_KA(len_secret_key);
            libbase::vector<bool> secret_key_KB(len_secret_key);

            print_bool_vector(
                "(prints from qkd_commsys.h)  Final Secret Key KA of Alice: ",
                secret_key_KA);
            print_bool_vector(
                "(prints from qkd_commsys.h)  Final Secret Key KB of Bob: ",
                secret_key_KB);

            return {std::move(secret_key_KA), std::move(secret_key_KA)};
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

    // Helper function to print a vector.
    void print_bool_vector(const std::string& title,
                           const libbase::vector<bool>& vec)
    {
        std::cout << "\n" << title << std::endl;
        for (int i = 0; i < vec.size(); ++i) {
            std::cout << vec(i) << "\t";
        }
        std::cout << std::endl;
    }

    // Serialization Support using shared pointers
    DECLARE_BASE_SERIALIZER(qkd_commsys)
    DECLARE_SERIALIZER(qkd_commsys)
};

} // namespace libcomm

#endif // __qkd_commsys_h